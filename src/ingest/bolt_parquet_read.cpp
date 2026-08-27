// bolt_parquet_read.cpp — W-PQ increment 2: Parquet page reader.
// See include/bolt/ingest/bolt_parquet_read.h for scope + safety contract.
//
// Notes:
//   - The RLE/bit-packed hybrid's bit-packed runs use LITTLE-ENDIAN bit
//     order in groups of 8 — the exact layout of bolt_pfor.h's
//     unpack_bits_scalar. We do NOT call that kernel here because it
//     requires 8 bytes of readable tail slack (kBitpackTailSlack) past
//     the payload; hostile parquet input gives no such guarantee, so
//     unpack_le_bounded below is the same loop with a bounds-checked
//     tail window instead of the slack contract.
//   - Per-page scratch (definition levels, dictionary indices) comes from
//     the caller Arena and is NOT recycled page-to-page (~8 bytes/row
//     worst case). Load-scoped arenas absorb this; revisit if a streaming
//     consumer ever feeds long-lived arenas through here.

#include "bolt/ingest/bolt_parquet_read.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/ingest/bolt_snappy.h"
#include "bolt/ingest/bolt_zstd_dec.h"
#include "bolt/ingest/bolt_inflate.h"
#include "bolt/ingest/bolt_lz4.h"
#include "bolt/ingest/bolt_lz4_raw.h"

namespace bolt {
namespace ingest {
namespace parquet {

namespace {

constexpr uint32_t kPqMaxPagesPerChunk = 1u << 16;
constexpr uint32_t kPqMaxDictEntries   = 1u << 24;
constexpr int64_t  kPqMaxPageBytes     = int64_t{1} << 30;
constexpr uint32_t kPqMetaChunksFirst  = 4096;

// parquet::PageType / Encoding / ConvertedType values we consume.
constexpr int32_t kPageData = 0, kPageIndex = 1, kPageDict = 2,
                  kPageDataV2 = 3;
constexpr int32_t kEncPlain = 0, kEncPlainDict = 2, kEncRle = 3,
                  kEncRleDict = 8, kEncByteStreamSplit = 9,
                  kEncDeltaBinaryPacked = 5, kEncDeltaLenByteArray = 6,
                  kEncDeltaByteArray = 7;
constexpr int32_t kConvDecimal = 5, kConvDate = 6;

inline void bit_clear(uint8_t* bm, int64_t i) noexcept {
    assert(bm != nullptr);
    assert(i >= 0);
    bm[i >> 3] &= static_cast<uint8_t>(~(1u << (i & 7)));
}

// ULEB128 over a bounded slice (the hybrid run headers).
bool uleb(const uint8_t* in, uint64_t in_len, uint64_t* ip,
          uint64_t* out) noexcept {
    assert(ip != nullptr && out != nullptr);
    uint64_t v = 0;
    for (uint32_t shift = 0; shift < 70; shift += 7) {   // bounded: 10 bytes
        if (*ip >= in_len) return false;
        const uint8_t b = in[(*ip)++];
        v |= static_cast<uint64_t>(b & 0x7Fu) << shift;
        if ((b & 0x80u) == 0) { *out = v; return true; }
    }
    return false;
}

// LSB-first bit unpack (bolt_pfor layout) with a bounds-checked tail
// window — see file header note.
void unpack_le_bounded(const uint8_t* BOLT_RESTRICT in, uint64_t in_len,
                       uint32_t n, uint32_t bw,
                       uint32_t* BOLT_RESTRICT out) noexcept {
    assert(in != nullptr || in_len == 0);
    assert(bw >= 1u && bw <= 32u);
    const uint64_t mask = (bw == 32u) ? 0xFFFFFFFFull : ((1ull << bw) - 1ull);

    // How many values can be read with an unconditional 8-byte load? The last
    // safe one is the largest i with (i*bw >> 3) + 8 <= in_len, i.e.
    // i*bw <= (in_len-8)*8 + 7. Solve it once instead of testing it per value.
    //
    // The per-value test was a branch on every one of 654M dictionary indices
    // in an SF10 lineitem decode (MEASURED), inside a loop whose whole body is
    // a load, a shift and a mask. It also blocked vectorisation: a loop with a
    // conditional load in it cannot be widened, so the tail-handling case cost
    // far more than the tail.
    uint32_t n_fast = 0;
    if (in_len >= 8u) {
        const uint64_t last = ((in_len - 8u) * 8u + 7u) / bw;
        n_fast = (last + 1u < static_cast<uint64_t>(n))
                     ? static_cast<uint32_t>(last + 1u)
                     : n;
    }
    assert(n_fast <= n);

    uint32_t i = 0;

    // Group-of-8 lane. Parquet bit-packs in groups of EIGHT, and eight values
    // of `bw` bits occupy exactly `bw` bytes (8*bw bits), so a group is always
    // byte-aligned and, for bw <= 8, fits in a single 64-bit load. Read the
    // group once and shift a running accumulator down by bw per value: no load
    // per value, and no variable shift either.
    //
    // The per-value loop below reads EIGHT BYTES FOR EVERY VALUE. At bw=3 that
    // is eight loads to extract the three bytes this lane loads once. Measured
    // width mix on SF10 lineitem dictionary indices: bw2 114M, bw3 180M,
    // bw4 120M, bw6 60M, bw12 180M -- so bw <= 8 covers 474M of 654M values.
    //
    // Technique from hardwood (dev.hardwood.internal.encoding.simd), which
    // structures its scalar unpacker the same way. bw > 8 stays on the general
    // loop: eight values would span up to 16 bytes and need a second load plus
    // cross-boundary handling, for the 28% of values at bw=12.
    if (bw <= 8u && in_len >= 8u) {
        const uint64_t max_group = (in_len - 8u) / bw;   // largest k: k*bw+8 <= in_len
        uint32_t groups = n / 8u;
        if (static_cast<uint64_t>(groups) > max_group + 1u) {
            groups = static_cast<uint32_t>(max_group + 1u);
        }
        for (uint32_t g = 0; g < groups; ++g) {
            uint64_t w;
            std::memcpy(&w, in + static_cast<uint64_t>(g) * bw, 8);
            out[i + 0] = static_cast<uint32_t>(w & mask); w >>= bw;
            out[i + 1] = static_cast<uint32_t>(w & mask); w >>= bw;
            out[i + 2] = static_cast<uint32_t>(w & mask); w >>= bw;
            out[i + 3] = static_cast<uint32_t>(w & mask); w >>= bw;
            out[i + 4] = static_cast<uint32_t>(w & mask); w >>= bw;
            out[i + 5] = static_cast<uint32_t>(w & mask); w >>= bw;
            out[i + 6] = static_cast<uint32_t>(w & mask); w >>= bw;
            out[i + 7] = static_cast<uint32_t>(w & mask);
            i += 8u;
        }
    }

    // Group-of-4 lane, for EVEN widths in 10..16. Four values span 4*bw bits
    // = bw/2 bytes, which is a whole number of bytes only when bw is even --
    // and 4*16 = 64 bits still fits one load. That reaches bw=12, which is 180M
    // of the 654M dictionary indices here (the other 28%), without the
    // cross-boundary handling a group of 8 would need at these widths.
    if (bw >= 10u && bw <= 16u && (bw & 1u) == 0u && in_len >= 8u) {
        const uint32_t half = bw / 2u;                   // bytes per 4 values
        const uint64_t max_group = (in_len - 8u) / half;
        uint32_t groups = (n - i) / 4u;
        if (static_cast<uint64_t>(groups) > max_group + 1u) {
            groups = static_cast<uint32_t>(max_group + 1u);
        }
        for (uint32_t g = 0; g < groups; ++g) {
            uint64_t w;
            std::memcpy(&w, in + static_cast<uint64_t>(g) * half, 8);
            out[i + 0] = static_cast<uint32_t>(w & mask); w >>= bw;
            out[i + 1] = static_cast<uint32_t>(w & mask); w >>= bw;
            out[i + 2] = static_cast<uint32_t>(w & mask); w >>= bw;
            out[i + 3] = static_cast<uint32_t>(w & mask);
            i += 4u;
        }
    }

    for (; i < n_fast; ++i) {          // no bounds test: proven above
        const uint64_t bit  = static_cast<uint64_t>(i) * bw;
        uint64_t w;
        std::memcpy(&w, in + (bit >> 3), 8);
        out[i] = static_cast<uint32_t>((w >> (bit & 7u)) & mask);
    }
    for (; i < n; ++i) {               // tail: fewer than 8 bytes remain
        const uint64_t bit  = static_cast<uint64_t>(i) * bw;
        const uint64_t byte = bit >> 3;
        uint8_t tmp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        assert(byte <= in_len);        // caller pre-checked total payload
        std::memcpy(tmp, in + byte, in_len - byte);
        uint64_t w = 0;
        std::memcpy(&w, tmp, 8);
        out[i] = static_cast<uint32_t>((w >> (bit & 7u)) & mask);
    }
}

// Parquet RLE/bit-packed hybrid: varint header; even => RLE run of
// (h>>1) copies of a ceil(bw/8)-byte LE value; odd => (h>>1) groups of
// 8 bit-packed values. Emission clamps at n (writers pad to groups).
bool rle_hybrid_decode(const uint8_t* in, uint64_t in_len, uint32_t bw,
                       uint32_t n, uint32_t* out) noexcept {
    assert(out != nullptr || n == 0);
    assert(bw <= 32u);
    if (n == 0) return true;
    if (bw == 0u) { std::memset(out, 0, size_t{n} * 4u); return true; }
    const uint32_t vbytes = (bw + 7u) / 8u;
    uint64_t ip = 0;
    uint32_t got = 0;
    while (got < n) {                  // bounded: every run consumes >=1 byte
        uint64_t h = 0;
        if (!uleb(in, in_len, &ip, &h)) return false;
        if ((h & 1u) == 0u) {          // RLE run
            const uint64_t count = h >> 1;
            if (count == 0 || ip + vbytes > in_len) return false;
            uint32_t v = 0;
            for (uint32_t k = 0; k < vbytes; ++k) {
                v |= static_cast<uint32_t>(in[ip + k]) << (8u * k);
            }
            ip += vbytes;
            if (bw < 32u) v &= (1u << bw) - 1u;
            uint64_t emit = count;
            if (emit > n - got) emit = n - got;
            for (uint64_t k = 0; k < emit; ++k) out[got + k] = v;
            got += static_cast<uint32_t>(emit);
        } else {                       // bit-packed groups of 8
            const uint64_t groups = h >> 1;
            const uint64_t bytes  = groups * bw;
            if (groups == 0 || ip + bytes > in_len) return false;
            uint64_t take = groups * 8u;
            if (take > n - got) take = n - got;
            unpack_le_bounded(in + ip, bytes, static_cast<uint32_t>(take),
                              bw, out + got);
            ip += bytes;
            got += static_cast<uint32_t>(take);
        }
    }
    assert(got == n);
    return true;
}

// Does a max_def_level==1 definition-level stream say "every value present",
// without materialising it?
//
// A nullable-but-null-free column is the common case, not a corner: a writer
// marks a column OPTIONAL from the schema, so a column with no nulls in it
// still carries a full definition-level stream of 1s. MEASURED on SF10
// lineitem, every one of the 959,776,832 definition levels (59,986,052 rows x
// 16 columns) arrives as an RLE RUN, and not one arrives bit-packed.
//
// decode_data_page already ends up on a dense path for these — it decodes the
// levels, sums them, finds nvalid == nvals and sets `def = nullptr`. The
// answer was never in doubt; the cost was reaching it. Expanding those runs
// writes 3.84 GB of uint32 that is then summed and discarded.
//
// So read the run headers and answer from them. Returns false for anything not
// provably all-present — a zero value, a bit-packed group, a short or
// malformed stream, a run set that does not cover the page — and the caller
// then does exactly what it did before. Fail-closed: this can only skip work it
// has proven unnecessary, never guess.
// Bits needed to hold a level in [0, max]. Parquet uses exactly this for both
// definition and repetition levels.
inline uint32_t level_bit_width(uint32_t max_lvl) noexcept {
    uint32_t w = 0;
    while ((1u << w) <= max_lvl) ++w;      // bounded: max_lvl <= 255 => w <= 8
    return (w == 0u) ? 1u : w;
}

bool rle_def_all_present(const uint8_t* in, uint64_t in_len, uint32_t n,
                         uint32_t max_def, uint32_t bw) noexcept {
    assert(in != nullptr || in_len == 0);
    if (n == 0) return false;                 // nothing to elide
    uint64_t ip = 0;
    uint32_t got = 0;
    while (got < n) {                  // bounded: every run consumes >=1 byte
        uint64_t h = 0;
        if (!uleb(in, in_len, &ip, &h)) return false;
        if ((h & 1u) != 0u) return false;     // bit-packed: caller decodes
        const uint64_t count = h >> 1;
        const uint64_t vbytes = (bw + 7u) / 8u;
        if (count == 0 || ip + vbytes > in_len) return false;
        uint32_t val = 0;
        for (uint64_t k = 0; k < vbytes; ++k) {
            val |= static_cast<uint32_t>(in[ip + k]) << (8u * k);
        }
        if (bw < 32u) val &= (1u << bw) - 1u;
        // Present means the level reached its MAXIMUM. For a flat optional
        // column that is 1; for a leaf inside an optional struct it is 2.
        if (val != max_def) return false;
        ip += vbytes;
        uint64_t emit = count;
        if (emit > n - got) emit = n - got;
        got += static_cast<uint32_t>(emit);
    }
    return got == n;
}

// Big-endian two's-complement FLBA -> 128-bit (hi, lo).
bool flba_to_i128(const uint8_t* p, uint32_t len, uint64_t* lo,
                  int64_t* hi) noexcept {
    assert(lo != nullptr && hi != nullptr);
    if (p == nullptr || len == 0 || len > 16) return false;
    const uint64_t fill = (p[0] & 0x80u) ? ~0ull : 0ull;
    uint64_t hi_u = fill, lo_u = fill;
    for (uint32_t i = 0; i < len; ++i) {     // bounded: len <= 16
        hi_u = (hi_u << 8) | (lo_u >> 56);
        lo_u = (lo_u << 8) | p[i];
    }
    *lo = lo_u;
    *hi = static_cast<int64_t>(hi_u);
    return true;
}

inline bool i128_fits_i64(uint64_t lo, int64_t hi) noexcept {
    return (hi == 0 && (lo >> 63) == 0u) || (hi == -1 && (lo >> 63) == 1u);
}

// ---- page header (thrift compact) ------------------------------------------

struct PageHdr {
    int32_t type;       // PageType
    int32_t unc;        // uncompressed_page_size
    int32_t cmp;        // compressed_page_size
    int32_t nvals;      // data/dict num_values
    int32_t enc;        // data/dict encoding
    int32_t def_enc;    // definition_level_encoding
    // DATA_PAGE_V2 only. V2 moves the levels OUT of the compressed body and
    // gives them explicit byte lengths, so they no longer have to be found by
    // decoding, and it drops v1's redundant 4-byte length prefix on each.
    int32_t def_len;    // definition_levels_byte_length
    int32_t rep_len;    // repetition_levels_byte_length
    uint8_t v2;         // 1 => DATA_PAGE_V2 layout
    uint8_t compressed; // v2's is_compressed (thrift default is TRUE)
};

// DataPageHeader {1 num_values, 2 encoding, 3 def_enc, 4 rep_enc, 5 stats}
bool parse_data_hdr(TcCursor* c, PageHdr* h) noexcept {
    assert(c != nullptr && h != nullptr);
    int16_t fid = 0;
    uint8_t ft;
    while (tc_field(c, &fid, &ft)) {
        int64_t v = 0;
        switch (fid) {
            case 1: if (!tc_zigzag(c, &v)) return false;
                    h->nvals = static_cast<int32_t>(v); break;
            case 2: if (!tc_zigzag(c, &v)) return false;
                    h->enc = static_cast<int32_t>(v); break;
            case 3: if (!tc_zigzag(c, &v)) return false;
                    h->def_enc = static_cast<int32_t>(v); break;
            default: if (!tc_skip(c, ft, 0)) return false; break;
        }
    }
    return true;
}

// DataPageHeaderV2 {1 num_values, 2 num_nulls, 3 num_rows, 4 encoding,
//                   5 def_levels_byte_length, 6 rep_levels_byte_length,
//                   7 is_compressed, 8 statistics}
bool parse_data_hdr_v2(TcCursor* c, PageHdr* h) noexcept {
    assert(c != nullptr && h != nullptr);
    h->v2 = 1;
    h->compressed = 1;            // thrift default when the field is absent
    h->def_enc = kEncRle;         // v2 levels are always RLE by spec
    int16_t fid = 0;
    uint8_t ft;
    while (tc_field(c, &fid, &ft)) {
        int64_t v = 0;
        switch (fid) {
            case 1: if (!tc_zigzag(c, &v)) return false;
                    h->nvals = static_cast<int32_t>(v); break;
            case 4: if (!tc_zigzag(c, &v)) return false;
                    h->enc = static_cast<int32_t>(v); break;
            case 5: if (!tc_zigzag(c, &v)) return false;
                    h->def_len = static_cast<int32_t>(v); break;
            case 6: if (!tc_zigzag(c, &v)) return false;
                    h->rep_len = static_cast<int32_t>(v); break;
            case 7: // compact protocol encodes bool in the field header type
                    h->compressed = (ft == 1u) ? 1u : 0u;
                    break;
            default: if (!tc_skip(c, ft, 0)) return false; break;
        }
    }
    return h->def_len >= 0 && h->rep_len >= 0;
}

// DictionaryPageHeader {1 num_values, 2 encoding, 3 is_sorted}
bool parse_dict_hdr(TcCursor* c, PageHdr* h) noexcept {
    assert(c != nullptr && h != nullptr);
    int16_t fid = 0;
    uint8_t ft;
    while (tc_field(c, &fid, &ft)) {
        int64_t v = 0;
        switch (fid) {
            case 1: if (!tc_zigzag(c, &v)) return false;
                    h->nvals = static_cast<int32_t>(v); break;
            case 2: if (!tc_zigzag(c, &v)) return false;
                    h->enc = static_cast<int32_t>(v); break;
            default: if (!tc_skip(c, ft, 0)) return false; break;
        }
    }
    return true;
}

// PageHeader {1 type, 2 unc, 3 cmp, 5 data_page_header,
//             7 dictionary_page_header, 8 data_page_header_v2} (4 crc skipped).
bool parse_page_header(TcCursor* c, PageHdr* h) noexcept {
    assert(c != nullptr && h != nullptr);
    h->type = -1; h->unc = -1; h->cmp = -1;
    h->nvals = -1; h->enc = -1; h->def_enc = -1;
    h->def_len = 0; h->rep_len = 0; h->v2 = 0; h->compressed = 1;
    int16_t fid = 0;
    uint8_t ft;
    while (tc_field(c, &fid, &ft)) {
        int64_t v = 0;
        switch (fid) {
            case 1: if (!tc_zigzag(c, &v)) return false;
                    h->type = static_cast<int32_t>(v); break;
            case 2: if (!tc_zigzag(c, &v)) return false;
                    h->unc = static_cast<int32_t>(v); break;
            case 3: if (!tc_zigzag(c, &v)) return false;
                    h->cmp = static_cast<int32_t>(v); break;
            case 5:
                if (ft != kTcStruct) return false;
                if (!parse_data_hdr(c, h)) return false;
                break;
            case 7:
                if (ft != kTcStruct) return false;
                if (!parse_dict_hdr(c, h)) return false;
                break;
            case 8:
                if (ft != kTcStruct) return false;
                if (!parse_data_hdr_v2(c, h)) return false;
                break;
            default: if (!tc_skip(c, ft, 0)) return false; break;
        }
    }
    if (h->type < 0 || h->unc < 0 || h->cmp < 0) return false;
    if (h->unc > kPqMaxPageBytes || h->cmp > kPqMaxPageBytes) return false;
    return true;
}

// ---- column decode context --------------------------------------------------

struct ColCtx {
    const PqColumn* pc;
    BoltType type;
    uint32_t elem;              // output stride: 1 / 2 / 4 / 8 / 16
    uint8_t* out;               // column data base
    uint8_t* validity;          // nullptr => no bitmap (all-valid column)
    char*    overflow;          // Utf8 spill buffer (column-wide)
    uint64_t overflow_cap;
    uint64_t overflow_cursor;
    uint8_t* dict;              // decoded dict entries, elem stride
    uint32_t dict_n;
    // G2FEAT-152 dictionary hint: per-row dictionary code, or nullptr when not
    // tracked. `codes_ok` starts 1 and is cleared by ANY page that is not
    // dictionary-encoded, so a chunk mixing PLAIN and dictionary pages emits no
    // hint rather than a partial one. -1 marks a null row.
    int32_t* codes;
    uint32_t codes_ok;
    // Type-conformance decode controls (G2FEAT-46).
    uint32_t src_width;         // physical source element width (bytes)
    uint8_t  src_signed;        // 1 => sign-extend on widen (INT32-DECIMAL /
                                //      signed ints); 0 => zero-extend (uints)
    uint8_t  int_convert;       // 1 => src_width != elem integer conversion
    uint8_t  is_int96;          // 1 => INT96 legacy timestamp -> Timestamp[us]
    int32_t  ts_rescale;        // 0 none / 1 millis*1000 / 2 nanos/1000 -> us
    // Utf8 spill growth (delta byte-array encodings only -- see
    // ensure_overflow_room). `ovf_base` points at the COLUMN's
    // str_overflow_base so a grow updates the view consumers read through.
    Arena*   arena;
    void**   ovf_base;
};

// Guarantee `need` more bytes of Utf8 spill room, growing if necessary.
//
// The spill buffer is sized from the chunk's UNCOMPRESSED page bytes. That is a
// valid bound for PLAIN byte arrays -- every value there carries a 4-byte
// length prefix, so uncompressed always exceeds materialised. It is NOT a bound
// for DELTA_BYTE_ARRAY or DELTA_LENGTH_BYTE_ARRAY: front coding means the page
// holds far fewer bytes than the strings it expands to, and DELTA_BYTE_ARRAY's
// expansion is not even linear in the page size, since each value may reuse an
// arbitrary prefix of its predecessor. MEASURED on a 31-row fixture: 527 bytes
// of text out of a 342-byte page, which failed with "OVERFLOW BUFFER FULL".
//
// Growing is safe because a StringView stores an OFFSET, not a pointer: moving
// the buffer and repointing str_overflow_base leaves every view already written
// still correct. The old block is left behind in the arena, which is bump
// allocated -- acceptable because this fires once per page at most, and only
// for encodings that need it.
inline bool ensure_overflow_room(ColCtx* cx, uint64_t need) noexcept {
    assert(cx != nullptr);
    if (cx->overflow_cursor + need <= cx->overflow_cap) return true;
    if (cx->arena == nullptr || cx->ovf_base == nullptr) return false;
    uint64_t cap = (cx->overflow_cap != 0u) ? cx->overflow_cap : 1024u;
    while (cap < cx->overflow_cursor + need) {      // bounded: doubles to fit
        if (cap > (1ull << 40)) return false;
        cap *= 2u;
    }
    char* nb = static_cast<char*>(cx->arena->allocate(cap, 1));
    if (nb == nullptr) return false;
    if (cx->overflow != nullptr && cx->overflow_cursor != 0u) {
        std::memcpy(nb, cx->overflow, cx->overflow_cursor);
    }
    cx->overflow = nb;
    cx->overflow_cap = cap;
    *cx->ovf_base = nb;
    return true;
}

// BOLT_PQ_DIAG=1 names WHICH bound a spilled-string decode hit. All three
// paths returned a bare false, reported upstream as an opaque decode
// failure. They are different problems: allocation, capacity sizing, and a
// format ceiling. Latched to one report per process.
bool bolt_pq_diag() noexcept {
    static const bool on = [] {
        const char* e = std::getenv("BOLT_PQ_DIAG");
        return e != nullptr && e[0] == '1';
    }();
    static bool fired = false;
    if (!on || fired) return false;
    fired = true;
    return true;
}

// Build a StringView; >12-byte payloads spill into the column overflow.
// Fixed-size moves. The size is a compile-time constant, so each of these
// lowers to one load/store pair -- never a call into libc's runtime
// size-dispatch ladder. Same reasoning dict_gather_dense already records
// below: the cost here is CALLS, not bytes.
inline void sv_copy8(uint8_t* d, const uint8_t* s) noexcept {
    uint64_t w; std::memcpy(&w, s, 8); std::memcpy(d, &w, 8);
}
inline void sv_copy4(uint8_t* d, const uint8_t* s) noexcept {
    uint32_t w; std::memcpy(&w, s, 4); std::memcpy(d, &w, 4);
}

// Copy exactly n bytes for n in [1,12] -- the inline StringView case.
//
// Overlapping fixed-width moves rather than one runtime-length memcpy: the
// pair covers every byte with no read past s+n and no write past d+n, so a
// caller holding only n readable bytes and a 12-byte destination is safe.
inline void sv_copy_small(uint8_t* d, const uint8_t* s, uint32_t n) noexcept {
    assert(n >= 1u && n <= 12u);
    if (n >= 8u) {                       // 8..12: two 8-byte moves, overlapping
        sv_copy8(d, s);
        sv_copy8(d + n - 8u, s + n - 8u);
    } else if (n >= 4u) {                // 4..7:  two 4-byte moves, overlapping
        sv_copy4(d, s);
        sv_copy4(d + n - 4u, s + n - 4u);
    } else {                             // 1..3:  too short to overlap-cover
        d[0] = s[0];
        if (n > 1u) d[1] = s[1];
        if (n > 2u) d[2] = s[2];
    }
}

// Copy n >= 8 bytes with 8-byte moves and an overlapping tail. Used for the
// spilled (>12 byte) string body, which on real data averages a few tens of
// bytes -- far too few to amortise a libc call, and there is one per ROW.
inline void sv_copy_bulk(uint8_t* d, const uint8_t* s, uint64_t n) noexcept {
    assert(n >= 8u);
    uint64_t k = 0;
    for (; k + 8u <= n; k += 8u) sv_copy8(d + k, s + k);   // bounded by n
    if (k < n) sv_copy8(d + n - 8u, s + n - 8u);           // exact tail
}

bool sv_from_bytes(ColCtx* cx, const uint8_t* p, uint32_t len,
                   StringView* out) noexcept {
    assert(cx != nullptr && out != nullptr);
    assert(p != nullptr || len == 0);
    std::memset(out, 0, sizeof(*out));
    out->length = len;
    if (len == 0) return true;
    if (len <= 12u) {
        // prefix[4] and inline_data[8] are contiguous by StringView's layout,
        // so the inline form is one 12-byte region starting at prefix.
        sv_copy_small(reinterpret_cast<uint8_t*>(out->prefix), p, len);
        return true;
    }
    if (cx->overflow == nullptr) {
        if (bolt_pq_diag())
            std::fprintf(stderr, "bolt parquet: NO OVERFLOW BUFFER for a "
                                 ">12-byte string (allocation failed)\n");
        return false;
    }
    const uint64_t cur = cx->overflow_cursor;
    if (cur + len > cx->overflow_cap) {
        if (bolt_pq_diag())
            std::fprintf(stderr, "bolt parquet: OVERFLOW BUFFER FULL -- "
                "cursor %llu + len %u > cap %llu (capacity, not the u32)\n",
                (unsigned long long)cur, len,
                (unsigned long long)cx->overflow_cap);
        return false;
    }
    if (cur > 0xFFFFFFFFull) {                   // StringView::ref.offset u32
        if (bolt_pq_diag())
            std::fprintf(stderr, "bolt parquet: U32 SPILL OFFSET EXCEEDED -- "
                "cursor %llu past 4 GB\n", (unsigned long long)cur);
        return false;
    }
    sv_copy4(reinterpret_cast<uint8_t*>(out->prefix), p);
    sv_copy_bulk(reinterpret_cast<uint8_t*>(cx->overflow) + cur, p, len);
    out->ref.buf_idx = 0;
    out->ref.offset  = static_cast<uint32_t>(cur);
    cx->overflow_cursor = cur + len;
    return true;
}

// ---- PLAIN value writers -----------------------------------------------------
// def == nullptr means dense (every row valid). With def, invalid rows
// clear their validity bit and consume no source value.

bool plain_fixed(ColCtx* cx, const uint8_t* v, uint64_t vlen,
                 const uint32_t* def, int64_t row0, uint32_t nrows,
                 uint32_t nvalid) noexcept {
    assert(cx != nullptr && cx->out != nullptr);
    const uint32_t w = cx->elem;
    assert(w == 4u || w == 8u);
    if (static_cast<uint64_t>(nvalid) * w > vlen) return false;
    uint8_t* dst = cx->out + static_cast<uint64_t>(row0) * w;
    if (def == nullptr) {
        std::memcpy(dst, v, static_cast<uint64_t>(nvalid) * w);
        return true;
    }
    uint64_t src = 0;
    for (uint32_t i = 0; i < nrows; ++i) {
        if (def[i] != cx->pc->max_def) { bit_clear(cx->validity, row0 + i); continue; }
        std::memcpy(dst + static_cast<uint64_t>(i) * w, v + src * w, w);
        ++src;
    }
    return src == nvalid;
}

// PLAIN FLOAT (4-byte f32 source) widened to Float64 output (elem 8).
// The source stride differs from the output stride, so plain_fixed's
// equal-width memcpy cannot be reused (G2FEAT-346).
bool plain_f32_widen(ColCtx* cx, const uint8_t* v, uint64_t vlen,
                     const uint32_t* def, int64_t row0, uint32_t nrows,
                     uint32_t nvalid) noexcept {
    assert(cx != nullptr && cx->out != nullptr);
    assert(cx->elem == 8u);
    if (static_cast<uint64_t>(nvalid) * 4u > vlen) return false;
    double* dst = reinterpret_cast<double*>(cx->out) + row0;
    uint64_t src = 0;
    for (uint32_t i = 0; i < nrows; ++i) {         // bounded: page rows
        if (def != nullptr && def[i] != cx->pc->max_def) {
            bit_clear(cx->validity, row0 + i);
            continue;
        }
        float f = 0.0f;
        std::memcpy(&f, v + src * 4u, 4);
        dst[i] = static_cast<double>(f);
        ++src;
    }
    return src == nvalid;
}

// PLAIN fixed integers of physical width `src_width` (4 or 8) converted to
// output width `elem` (1/2/4/8), value-preserving (G2FEAT-46). Narrowing
// drops the high bytes (LE host, as the rest of this codec assumes);
// widening sign-extends when `src_signed` (INT32-DECIMAL mantissa -> int64
// Decimal64; signed sub-int) or zero-extends for unsigned ints. Handles
// INT32-DECIMAL, UINT_8/16, INT_8/16 on the data page AND the dict page.
bool plain_int_conv(ColCtx* cx, const uint8_t* v, uint64_t vlen,
                    const uint32_t* def, int64_t row0, uint32_t nrows,
                    uint32_t nvalid) noexcept {
    assert(cx != nullptr && cx->out != nullptr);
    const uint32_t sw = cx->src_width;
    const uint32_t dw = cx->elem;
    assert((sw == 4u || sw == 8u) && dw >= 1u && dw <= 8u);
    if (static_cast<uint64_t>(nvalid) * sw > vlen) return false;
    uint64_t src = 0;
    for (uint32_t i = 0; i < nrows; ++i) {         // bounded: page rows
        if (def != nullptr && def[i] != cx->pc->max_def) {
            bit_clear(cx->validity, row0 + i);
            continue;
        }
        int64_t val = 0;
        if (sw == 4u) {
            int32_t t = 0;
            std::memcpy(&t, v + src * 4u, 4);
            val = cx->src_signed ? static_cast<int64_t>(t)
                                 : static_cast<int64_t>(static_cast<uint32_t>(t));
        } else {
            std::memcpy(&val, v + src * 8u, 8);
        }
        ++src;
        std::memcpy(cx->out + (static_cast<uint64_t>(row0) + i) * dw, &val, dw);
    }
    return src == nvalid;
}

// INT96 legacy timestamp (12 bytes: 8-byte nanoseconds-within-day LE +
// 4-byte Julian day LE) -> Timestamp[us] output (elem 8). Impala/Spark
// wrote this before the INT64 logical timestamp (G2FEAT-46).
bool plain_int96(ColCtx* cx, const uint8_t* v, uint64_t vlen,
                 const uint32_t* def, int64_t row0, uint32_t nrows,
                 uint32_t nvalid) noexcept {
    assert(cx != nullptr && cx->out != nullptr);
    assert(cx->elem == 8u);
    if (static_cast<uint64_t>(nvalid) * 12u > vlen) return false;
    int64_t* dst = reinterpret_cast<int64_t*>(cx->out) + row0;
    uint64_t src = 0;
    for (uint32_t i = 0; i < nrows; ++i) {         // bounded: page rows
        if (def != nullptr && def[i] != cx->pc->max_def) {
            bit_clear(cx->validity, row0 + i);
            continue;
        }
        const uint8_t* s = v + src * 12u;
        uint64_t nanos_of_day = 0;
        int32_t  julian = 0;
        std::memcpy(&nanos_of_day, s, 8);
        std::memcpy(&julian, s + 8, 4);
        const int64_t days = static_cast<int64_t>(julian) - 2440588;  // epoch
        dst[i] = days * 86400000000LL +
                 static_cast<int64_t>(nanos_of_day / 1000ull);
        ++src;
    }
    return src == nvalid;
}

// PLAIN BOOLEAN: bit-packed LSB-first, one bit per VALID value ->
// Int64 0/1 output (elem 8). Invalid rows consume no source bit.
bool plain_bool(ColCtx* cx, const uint8_t* v, uint64_t vlen,
                const uint32_t* def, int64_t row0, uint32_t nrows,
                uint32_t nvalid) noexcept {
    assert(cx != nullptr && cx->out != nullptr);
    assert(cx->elem == 8u);
    if ((static_cast<uint64_t>(nvalid) + 7u) / 8u > vlen) return false;
    int64_t* dst = reinterpret_cast<int64_t*>(cx->out) + row0;
    uint64_t src = 0;
    for (uint32_t i = 0; i < nrows; ++i) {         // bounded: page rows
        if (def != nullptr && def[i] != cx->pc->max_def) {
            bit_clear(cx->validity, row0 + i);
            continue;
        }
        dst[i] = static_cast<int64_t>((v[src >> 3] >> (src & 7u)) & 1u);
        ++src;
    }
    return src == nvalid;
}

// RLE-encoded BOOLEAN data page (Encoding::RLE is legal for data values
// on BOOLEAN only): u32 LE byte length, then an RLE/bit-packed hybrid
// stream of bit-width-1 values -> Int64 0/1 output, def-level aware.
bool rle_bool(ColCtx* cx, const uint8_t* v, uint64_t vlen,
              const uint32_t* def, int64_t row0, uint32_t nrows,
              uint32_t nvalid, Arena* arena) noexcept {
    assert(cx != nullptr && cx->out != nullptr);
    assert(cx->elem == 8u);
    assert(arena != nullptr);
    uint32_t* bits = nullptr;
    if (nvalid > 0) {
        if (vlen < 4) return false;
        uint32_t bl = 0;
        std::memcpy(&bl, v, 4);
        if (4ull + bl > vlen) return false;
        bits = static_cast<uint32_t*>(
            arena->allocate(uint64_t{nvalid} * 4u, 4));
        if (bits == nullptr) return false;
        if (!rle_hybrid_decode(v + 4, bl, 1, nvalid, bits)) return false;
    }
    int64_t* dst = reinterpret_cast<int64_t*>(cx->out) + row0;
    uint32_t k = 0;
    for (uint32_t i = 0; i < nrows; ++i) {         // bounded: page rows
        if (def != nullptr && def[i] != cx->pc->max_def) {
            bit_clear(cx->validity, row0 + i);
            continue;
        }
        if (k >= nvalid) return false;
        dst[i] = static_cast<int64_t>(bits[k++] & 1u);
    }
    return k == nvalid;
}

// FLBA DECIMAL -> Decimal64 mantissa (elem 8) or Decimal128 (elem 16).
bool plain_flba(ColCtx* cx, const uint8_t* v, uint64_t vlen,
                const uint32_t* def, int64_t row0, uint32_t nrows,
                uint32_t nvalid) noexcept {
    assert(cx != nullptr && cx->out != nullptr);
    assert(cx->elem == 8u || cx->elem == 16u);
    const uint32_t fw = static_cast<uint32_t>(cx->pc->type_length);
    if (fw == 0u || fw > 16u) return false;
    if (static_cast<uint64_t>(nvalid) * fw > vlen) return false;
    uint64_t src = 0;
    for (uint32_t i = 0; i < nrows; ++i) {
        if (def != nullptr && def[i] != cx->pc->max_def) {
            bit_clear(cx->validity, row0 + i);
            continue;
        }
        uint64_t lo = 0;
        int64_t  hi = 0;
        if (!flba_to_i128(v + src * fw, fw, &lo, &hi)) return false;
        ++src;
        uint8_t* dst =
            cx->out + (static_cast<uint64_t>(row0) + i) * cx->elem;
        if (cx->elem == 8u) {
            if (!i128_fits_i64(lo, hi)) return false;   // precision lied
            std::memcpy(dst, &lo, 8);
        } else {                                        // Decimal128 {lo, hi}
            std::memcpy(dst, &lo, 8);
            std::memcpy(dst + 8, &hi, 8);
        }
    }
    return src == nvalid;
}

// BYTE_ARRAY (u32 LE length prefix per value) -> Utf8 StringViews.
bool plain_utf8(ColCtx* cx, const uint8_t* v, uint64_t vlen,
                const uint32_t* def, int64_t row0, uint32_t nrows,
                uint32_t nvalid) noexcept {
    assert(cx != nullptr && cx->out != nullptr);
    assert(cx->elem == 16u);
    auto* views = reinterpret_cast<StringView*>(cx->out);
    uint64_t pos = 0;
    uint32_t src = 0;
    for (uint32_t i = 0; i < nrows; ++i) {
        if (def != nullptr && def[i] != cx->pc->max_def) {
            bit_clear(cx->validity, row0 + i);
            continue;
        }
        if (pos + 4 > vlen) return false;
        uint32_t blen = 0;
        std::memcpy(&blen, v + pos, 4);
        pos += 4;
        if (pos + blen > vlen) return false;
        if (!sv_from_bytes(cx, v + pos, blen, &views[row0 + i])) return false;
        pos += blen;
        ++src;
    }
    return src == nvalid;
}

// Gather dictionary entries (already in output representation).
//
// The element width is a COMPILE-TIME parameter on purpose. Written as
// `memcpy(dst, src, cx->elem)` with a runtime width, the compiler cannot see
// the size and emits a call to libc `memmove` for every single value; on
// ClickBench `hits` (99,997,497 rows) that call was measured at ~34% of ALL
// decode CPU — more than snappy and the RLE unpacker combined — to move four
// bytes a hundred million times. With `W` constant the same `memcpy` lowers to
// one load/store pair and no call at all. `memcpy` (not a typed `T*` store) is
// deliberate: it keeps the gather alignment-agnostic and free of any
// strict-aliasing assumption about the arena buffers.
template <uint32_t W>
void dict_gather_dense(uint8_t* BOLT_RESTRICT out,
                       const uint8_t* BOLT_RESTRICT dict,
                       const uint32_t* BOLT_RESTRICT idx,
                       uint32_t n) noexcept {
    assert(out != nullptr || n == 0);
    assert(dict != nullptr || n == 0);
    for (uint32_t i = 0; i < n; ++i) {
        std::memcpy(out + static_cast<uint64_t>(i) * W,
                    dict + static_cast<uint64_t>(idx[i]) * W, W);
    }
}

// Dense gather that ALSO records the code hint.
//
// A dictionary Utf8 column carries the G2FEAT-152 code hint, and that alone
// used to send it to dict_gather_sparse even when the page has no nulls at
// all -- so every row paid a null test and a `k >= nvalid` bound test to
// support a null that was not there. On SF10 lineitem the four
// dictionary-encoded Utf8 columns are 258.0 ms; forcing them down the plain
// dense path (which drops the code hint, so it is not a legal fix, only a
// ceiling) gives 174.5 ms. This recovers most of that gap while still
// recording codes: dense means k == i, so both tests go.
template <uint32_t W>
void dict_gather_dense_codes(uint8_t* BOLT_RESTRICT out,
                             const uint8_t* BOLT_RESTRICT dict,
                             const uint32_t* BOLT_RESTRICT idx,
                             int32_t* BOLT_RESTRICT codes,
                             uint32_t n) noexcept {
    assert(out != nullptr || n == 0);
    assert(dict != nullptr || n == 0);
    assert(codes != nullptr || n == 0);
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t id = idx[i];       // range pre-validated by the caller
        codes[i] = static_cast<int32_t>(id);
        std::memcpy(out + static_cast<uint64_t>(i) * W,
                    dict + static_cast<uint64_t>(id) * W, W);
    }
}

// Nullable / code-tracking gather: one row at a time, but still a
// compile-time-width copy. Reached only by an OPTIONAL column that actually
// contains nulls -- a dense page with a code hint now takes
// dict_gather_dense_codes above.
template <uint32_t W>
bool dict_gather_sparse(ColCtx* cx, const uint32_t* idx, const uint32_t* def,
                        int64_t row0, uint32_t nrows,
                        uint32_t nvalid) noexcept {
    assert(cx != nullptr && cx->out != nullptr);
    assert(cx->dict != nullptr);
    uint32_t k = 0;
    for (uint32_t i = 0; i < nrows; ++i) {
        if (def != nullptr && def[i] != cx->pc->max_def) {
            bit_clear(cx->validity, row0 + i);
            if (cx->codes != nullptr) cx->codes[row0 + i] = -1;   // null row
            continue;
        }
        if (k >= nvalid) return false;
        const uint32_t id = idx[k++];          // range pre-validated by caller
        if (cx->codes != nullptr) {
            cx->codes[row0 + i] = static_cast<int32_t>(id);
        }
        std::memcpy(cx->out + (static_cast<uint64_t>(row0) + i) * W,
                    cx->dict + static_cast<uint64_t>(id) * W, W);
    }
    return k == nvalid;
}

bool dict_gather(ColCtx* cx, const uint32_t* idx, const uint32_t* def,
                 int64_t row0, uint32_t nrows, uint32_t nvalid) noexcept {
    assert(cx != nullptr && cx->out != nullptr);
    assert(idx != nullptr || nvalid == 0);
    if (cx->dict == nullptr) return false;
    const uint32_t w = cx->elem;

    // Validate the whole index block ONCE with a max-reduction the compiler
    // vectorizes, instead of an `id >= dict_n` branch per value. Every index
    // the gather below can consume lies in [0, nvalid), so this is a superset
    // of the old per-value check: any page the old code accepted is accepted
    // here with byte-identical output, and a corrupt page is now rejected
    // BEFORE any row is written rather than partway through.
    uint32_t mx = 0;
    for (uint32_t k = 0; k < nvalid; ++k) mx = (idx[k] > mx) ? idx[k] : mx;
    if (nvalid > 0 && mx >= cx->dict_n) return false;

    // Dense fast path: no nulls to skip, so the gather is a pure permutation
    // of `nrows` values. Whether a code hint has to be recorded alongside
    // decides WHICH dense gather, not whether one is possible -- a dense page
    // with codes used to fall all the way through to the sparse gather.
    if (def == nullptr) {
        if (nrows != nvalid) return false;   // dense page: every row valid
        uint8_t* out = cx->out + static_cast<uint64_t>(row0) * w;
        if (cx->codes == nullptr) {
            switch (w) {
                case 8:  dict_gather_dense<8>(out, cx->dict, idx, nrows);  return true;
                case 4:  dict_gather_dense<4>(out, cx->dict, idx, nrows);  return true;
                case 16: dict_gather_dense<16>(out, cx->dict, idx, nrows); return true;
                case 2:  dict_gather_dense<2>(out, cx->dict, idx, nrows);  return true;
                case 1:  dict_gather_dense<1>(out, cx->dict, idx, nrows);  return true;
                default: return false;
            }
        }
        int32_t* codes = cx->codes + row0;
        switch (w) {
            case 16: dict_gather_dense_codes<16>(out, cx->dict, idx, codes, nrows); return true;
            case 8:  dict_gather_dense_codes<8>(out, cx->dict, idx, codes, nrows);  return true;
            case 4:  dict_gather_dense_codes<4>(out, cx->dict, idx, codes, nrows);  return true;
            case 2:  dict_gather_dense_codes<2>(out, cx->dict, idx, codes, nrows);  return true;
            case 1:  dict_gather_dense_codes<1>(out, cx->dict, idx, codes, nrows);  return true;
            default: return false;
        }
    }
    switch (w) {
        case 8:  return dict_gather_sparse<8>(cx, idx, def, row0, nrows, nvalid);
        case 4:  return dict_gather_sparse<4>(cx, idx, def, row0, nrows, nvalid);
        case 16: return dict_gather_sparse<16>(cx, idx, def, row0, nrows, nvalid);
        case 2:  return dict_gather_sparse<2>(cx, idx, def, row0, nrows, nvalid);
        case 1:  return dict_gather_sparse<1>(cx, idx, def, row0, nrows, nvalid);
        default: return false;
    }
}

// ---- pages -------------------------------------------------------------------

// Dictionary page: PLAIN-encoded values, converted ONCE into the output
// representation so data-page gathers are bare memcpys.
bool decode_dict_page(ColCtx* cx, const uint8_t* v, uint64_t vlen,
                      int32_t nvals, Arena* arena) noexcept {
    assert(cx != nullptr && arena != nullptr);
    if (cx->dict != nullptr) return false;          // two dict pages: corrupt
    if (nvals < 0 || static_cast<uint32_t>(nvals) > kPqMaxDictEntries) {
        return false;
    }
    const uint32_t n = static_cast<uint32_t>(nvals);
    uint8_t* dict = static_cast<uint8_t*>(
        arena->allocate(static_cast<uint64_t>(n) * cx->elem + 16u, 16));
    if (dict == nullptr) return false;
    // Reuse the PLAIN writers against a temporary dense "column" view.
    ColCtx tmp = *cx;
    tmp.out = dict;
    tmp.validity = nullptr;
    bool ok = false;
    switch (cx->pc->physical) {
        case PqType::Int32:
        case PqType::Int64:
            // Convert dict entries to the OUTPUT representation once (e.g.
            // INT32-DECIMAL mantissa sign-extended to Decimal64, unsigned
            // sub-int narrowed) so data-page gathers stay bare memcpys.
            ok = cx->int_convert
                     ? plain_int_conv(&tmp, v, vlen, nullptr, 0, n, n)
                     : plain_fixed(&tmp, v, vlen, nullptr, 0, n, n);
            break;
        case PqType::Double:
            ok = plain_fixed(&tmp, v, vlen, nullptr, 0, n, n);
            break;
        case PqType::Int96:
            // 12-byte legacy timestamps -> Timestamp[us] once (G2FEAT-46).
            ok = plain_int96(&tmp, v, vlen, nullptr, 0, n, n);
            break;
        case PqType::Float:
            // Dict entries widened f32 -> f64 ONCE here so data-page
            // gathers stay bare 8-byte memcpys (G2FEAT-346).
            ok = plain_f32_widen(&tmp, v, vlen, nullptr, 0, n, n);
            break;
        case PqType::FixedLenByteArray:
            ok = plain_flba(&tmp, v, vlen, nullptr, 0, n, n);
            break;
        case PqType::ByteArray:
            ok = plain_utf8(&tmp, v, vlen, nullptr, 0, n, n);
            break;
        default:
            ok = false;
            break;
    }
    if (!ok) return false;
    cx->overflow_cursor = tmp.overflow_cursor;      // Utf8 spill advanced
    cx->dict = dict;
    cx->dict_n = n;
    return true;
}



bool decode_plain_values(ColCtx* cx, const uint8_t* v, uint64_t vlen,
                         const uint32_t* def, int64_t row0, uint32_t nvals,
                         uint32_t nvalid) noexcept;

// Zig-zag: the spec stores signed deltas so that small magnitudes of either
// sign use few bits.
inline int64_t zigzag_decode(uint64_t n) noexcept {
    return static_cast<int64_t>(n >> 1) ^ -static_cast<int64_t>(n & 1u);
}

// Bit-unpack to 64 bits. unpack_le_bounded tops out at bw <= 32, but a
// DELTA_BINARY_PACKED miniblock over int64 data needs up to 64 -- a column of
// random int64s has deltas near 2^41, which is already past 32. Values can
// straddle a 64-bit word, so this reads two words; the tail copies into a
// zero-filled 16-byte window rather than reading past the page.
void unpack_le64(const uint8_t* BOLT_RESTRICT in, uint64_t in_len, uint32_t n,
                 uint32_t bw, uint64_t* BOLT_RESTRICT out) noexcept {
    assert(bw >= 1u && bw <= 64u);
    const uint64_t mask = (bw == 64u) ? ~0ull : ((1ull << bw) - 1ull);
    for (uint32_t i = 0; i < n; ++i) {
        const uint64_t bit  = static_cast<uint64_t>(i) * bw;
        const uint64_t byte = bit >> 3;
        const uint32_t sh   = static_cast<uint32_t>(bit & 7u);
        uint64_t lo = 0, hi = 0;
        if (byte + 16u <= in_len) {
            std::memcpy(&lo, in + byte, 8);
            std::memcpy(&hi, in + byte + 8, 8);
        } else {
            uint8_t tmp[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
            const uint64_t avail = (byte < in_len) ? (in_len - byte) : 0;
            std::memcpy(tmp, in + byte, (avail < 16u) ? avail : 16u);
            std::memcpy(&lo, tmp, 8);
            std::memcpy(&hi, tmp + 8, 8);
        }
        uint64_t val = lo >> sh;
        if (sh != 0u) val |= hi << (64u - sh);
        out[i] = val & mask;
    }
}

// DELTA_BINARY_PACKED (encoding 5) -- the INT32/INT64 encoding of the parquet
// V2 writer, so Spark with writer.version=v2 emits it for every integer column.
//
//   header: block_size (values, multiple of 128), miniblocks_per_block,
//           total_value_count, first_value (zig-zag)
//   block:  min_delta (zig-zag), one bit-width byte per miniblock,
//           then each miniblock's deltas bit-packed at its own width
//
// value[0] = first_value;  value[i] = value[i-1] + min_delta + unpacked[i]
//
// Decoded into a PLAIN-layout buffer at the physical width and handed to
// decode_plain_values, so the logical-type and widening rules are inherited
// rather than restated.
// Decode `want` values from a DELTA_BINARY_PACKED block into int64, reporting
// how many bytes the block consumed. Split out of delta_binary_packed because
// DELTA_LENGTH_BYTE_ARRAY and DELTA_BYTE_ARRAY encode their length arrays this
// way and then place raw bytes immediately after -- so the consumed count is
// what locates the data.
bool delta_decode_ints(const uint8_t* v, uint64_t vlen, uint32_t want,
                       int64_t* out, uint64_t* consumed, Arena* arena) noexcept {
    assert(out != nullptr || want == 0);
    assert(consumed != nullptr && arena != nullptr);
    uint64_t ip = 0, block_size = 0, n_mini = 0, total = 0, first_zz = 0;
    if (!uleb(v, vlen, &ip, &block_size)) return false;
    if (!uleb(v, vlen, &ip, &n_mini)) return false;
    if (!uleb(v, vlen, &ip, &total)) return false;
    if (!uleb(v, vlen, &ip, &first_zz)) return false;
    if (block_size == 0u || n_mini == 0u || n_mini > 64u) return false;
    if ((block_size % n_mini) != 0u) return false;
    const uint64_t vpm = block_size / n_mini;
    if (vpm == 0u || (vpm % 8u) != 0u) return false;      // spec: multiple of 8
    if (total < want) return false;
    if (want == 0u) { *consumed = ip; return true; }

    auto* tmp = static_cast<uint64_t*>(arena->allocate(vpm * 8u, 8));
    if (tmp == nullptr) return false;
    uint64_t produced = 0;
    int64_t cur = zigzag_decode(first_zz);
    out[produced++] = cur;
    while (produced < want) {        // bounded: every block consumes >= 1 byte
        uint64_t min_zz = 0;
        if (!uleb(v, vlen, &ip, &min_zz)) return false;
        const int64_t min_delta = zigzag_decode(min_zz);
        if (ip + n_mini > vlen) return false;
        const uint8_t* widths = v + ip;
        ip += n_mini;
        for (uint64_t m = 0; m < n_mini && produced < want; ++m) {
            const uint32_t bw = widths[m];
            if (bw > 64u) return false;
            const uint64_t bytes = (vpm * bw) / 8u;
            if (ip + bytes > vlen) return false;
            if (bw == 0u) {
                for (uint64_t k = 0; k < vpm && produced < want; ++k) {
                    cur += min_delta;
                    out[produced++] = cur;
                }
            } else {
                unpack_le64(v + ip, vlen - ip, static_cast<uint32_t>(vpm), bw, tmp);
                for (uint64_t k = 0; k < vpm && produced < want; ++k) {
                    cur += min_delta + static_cast<int64_t>(tmp[k]);
                    out[produced++] = cur;
                }
            }
            ip += bytes;
        }
    }
    *consumed = ip;
    return true;
}

// DELTA_BINARY_PACKED (encoding 5) -- the INT32/INT64 encoding of the parquet
// V2 writer, so Spark with writer.version=v2 emits it for every integer column.
// Decoded into a PLAIN-layout buffer at the physical width and handed to
// decode_plain_values, so logical-type and widening rules are inherited.
bool delta_binary_packed(ColCtx* cx, const uint8_t* v, uint64_t vlen,
                         const uint32_t* def, int64_t row0, uint32_t nvals,
                         uint32_t nvalid, Arena* arena) noexcept {
    assert(cx != nullptr && arena != nullptr);
    const PqType phys = cx->pc->physical;
    if (phys != PqType::Int32 && phys != PqType::Int64) return false;
    const uint32_t w = (phys == PqType::Int32) ? 4u : 8u;
    if (nvalid == 0u) return decode_plain_values(cx, v, 0, def, row0, nvals, 0);

    auto* vals = static_cast<int64_t*>(
        arena->allocate(static_cast<uint64_t>(nvalid) * 8u, 8));
    auto* outbuf = static_cast<uint8_t*>(
        arena->allocate(static_cast<uint64_t>(nvalid) * w, 8));
    if (vals == nullptr || outbuf == nullptr) return false;
    uint64_t used = 0;
    if (!delta_decode_ints(v, vlen, nvalid, vals, &used, arena)) return false;
    for (uint32_t i = 0; i < nvalid; ++i) {
        if (w == 8u) {
            std::memcpy(outbuf + static_cast<uint64_t>(i) * 8u, &vals[i], 8);
        } else {
            // Narrowing is intentional: the spec's arithmetic is modular.
            const int32_t t = static_cast<int32_t>(vals[i]);
            std::memcpy(outbuf + static_cast<uint64_t>(i) * 4u, &t, 4);
        }
    }
    return decode_plain_values(cx, outbuf, static_cast<uint64_t>(nvalid) * w,
                               def, row0, nvals, nvalid);
}

// DELTA_LENGTH_BYTE_ARRAY (encoding 6): a delta-packed length array, then all
// the bytes concatenated. Rebuilt into the PLAIN byte-array layout (u32 length
// then bytes, per value) and handed to the PLAIN decoder, so the Utf8 spill
// handling and null skipping are inherited rather than restated.
bool delta_length_byte_array(ColCtx* cx, const uint8_t* v, uint64_t vlen,
                             const uint32_t* def, int64_t row0, uint32_t nvals,
                             uint32_t nvalid, Arena* arena) noexcept {
    assert(cx != nullptr && arena != nullptr);
    if (cx->pc->physical != PqType::ByteArray) return false;
    if (nvalid == 0u) return decode_plain_values(cx, v, 0, def, row0, nvals, 0);
    auto* lens = static_cast<int64_t*>(
        arena->allocate(static_cast<uint64_t>(nvalid) * 8u, 8));
    if (lens == nullptr) return false;
    uint64_t used = 0;
    if (!delta_decode_ints(v, vlen, nvalid, lens, &used, arena)) return false;

    uint64_t total = 0;
    for (uint32_t i = 0; i < nvalid; ++i) {
        if (lens[i] < 0) return false;
        total += static_cast<uint64_t>(lens[i]);
    }
    if (used + total > vlen) return false;
    const uint8_t* data = v + used;
    if (!ensure_overflow_room(cx, total)) return false;
    const uint64_t plain_len = total + static_cast<uint64_t>(nvalid) * 4u;
    auto* plain = static_cast<uint8_t*>(arena->allocate(plain_len, 8));
    if (plain == nullptr) return false;
    uint64_t sp = 0, dp = 0;
    for (uint32_t i = 0; i < nvalid; ++i) {
        const uint32_t l = static_cast<uint32_t>(lens[i]);
        std::memcpy(plain + dp, &l, 4);
        dp += 4;
        std::memcpy(plain + dp, data + sp, l);
        dp += l;
        sp += l;
    }
    return decode_plain_values(cx, plain, plain_len, def, row0, nvals, nvalid);
}

// DELTA_BYTE_ARRAY (encoding 7): incremental (front-coded) strings -- a
// delta-packed prefix-length array, a delta-packed suffix-length array, then
// the suffix bytes. Each value reuses the first `prefix[i]` bytes of its
// PREDECESSOR, which is why sorted string columns compress so well with it.
bool delta_byte_array(ColCtx* cx, const uint8_t* v, uint64_t vlen,
                      const uint32_t* def, int64_t row0, uint32_t nvals,
                      uint32_t nvalid, Arena* arena) noexcept {
    assert(cx != nullptr && arena != nullptr);
    if (cx->pc->physical != PqType::ByteArray) return false;
    if (nvalid == 0u) return decode_plain_values(cx, v, 0, def, row0, nvals, 0);

    auto* pre = static_cast<int64_t*>(
        arena->allocate(static_cast<uint64_t>(nvalid) * 8u, 8));
    auto* suf = static_cast<int64_t*>(
        arena->allocate(static_cast<uint64_t>(nvalid) * 8u, 8));
    if (pre == nullptr || suf == nullptr) return false;
    uint64_t used_p = 0, used_s = 0;
    if (!delta_decode_ints(v, vlen, nvalid, pre, &used_p, arena)) return false;
    if (!delta_decode_ints(v + used_p, vlen - used_p, nvalid, suf, &used_s,
                           arena)) {
        return false;
    }
    const uint64_t hdr = used_p + used_s;
    if (hdr > vlen) return false;
    const uint8_t* data = v + hdr;
    const uint64_t data_len = vlen - hdr;

    // Bound the output before writing any of it: value i is
    // prefix[i] + suffix[i] long, and prefix[i] may not exceed the length of
    // value i-1 (a value cannot borrow bytes its predecessor does not have).
    uint64_t total = 0, prev_len = 0, sufsum = 0;
    for (uint32_t i = 0; i < nvalid; ++i) {
        if (pre[i] < 0 || suf[i] < 0) return false;
        const uint64_t p = static_cast<uint64_t>(pre[i]);
        const uint64_t sfx = static_cast<uint64_t>(suf[i]);
        if (p > prev_len) return false;             // corrupt back-reference
        sufsum += sfx;
        if (sufsum > data_len) return false;
        prev_len = p + sfx;
        total += prev_len;
    }
    if (!ensure_overflow_room(cx, total)) return false;
    const uint64_t plain_len = total + static_cast<uint64_t>(nvalid) * 4u;
    auto* plain = static_cast<uint8_t*>(arena->allocate(plain_len, 8));
    if (plain == nullptr) return false;

    // `last` points at the previous value inside `plain`, so the prefix copy
    // reads bytes already materialised -- no second scratch buffer.
    uint64_t dp = 0, sp = 0, last_off = 0, last_len = 0;
    for (uint32_t i = 0; i < nvalid; ++i) {
        const uint32_t p = static_cast<uint32_t>(pre[i]);
        const uint32_t sfx = static_cast<uint32_t>(suf[i]);
        const uint32_t l = p + sfx;
        std::memcpy(plain + dp, &l, 4);
        dp += 4;
        if (p != 0u) std::memcpy(plain + dp, plain + last_off, p);
        std::memcpy(plain + dp + p, data + sp, sfx);
        last_off = dp;
        last_len = l;
        (void)last_len;
        dp += l;
        sp += sfx;
    }
    return decode_plain_values(cx, plain, plain_len, def, row0, nvals, nvalid);
}

// PLAIN value dispatch, by physical type. Lifted out of decode_data_page so
// BYTE_STREAM_SPLIT can hand it a transposed buffer and inherit every width
// and logical-type conversion rather than restating them.
bool decode_plain_values(ColCtx* cx, const uint8_t* v, uint64_t vlen,
                         const uint32_t* def, int64_t row0, uint32_t nvals,
                         uint32_t nvalid) noexcept {
    assert(cx != nullptr);
    if (cx->is_int96) {
        return plain_int96(cx, v, vlen, def, row0, nvals, nvalid);
    }
    switch (cx->pc->physical) {
        case PqType::Int32:
        case PqType::Int64:
            return cx->int_convert
                       ? plain_int_conv(cx, v, vlen, def, row0, nvals, nvalid)
                       : plain_fixed(cx, v, vlen, def, row0, nvals, nvalid);
        case PqType::Double:
            return plain_fixed(cx, v, vlen, def, row0, nvals, nvalid);
        case PqType::Float:
            return plain_f32_widen(cx, v, vlen, def, row0, nvals, nvalid);
        case PqType::Boolean:
            return plain_bool(cx, v, vlen, def, row0, nvals, nvalid);
        case PqType::FixedLenByteArray:
            return plain_flba(cx, v, vlen, def, row0, nvals, nvalid);
        case PqType::ByteArray:
            return plain_utf8(cx, v, vlen, def, row0, nvals, nvalid);
        default:
            return false;
    }
}

// BYTE_STREAM_SPLIT (encoding 9).
//
// The page holds W byte-streams laid end to end: every value's byte 0, then
// every value's byte 1, and so on. Value i's byte j lives at in[j*N + i]. The
// point of the encoding is that a column of similar floats has low entropy in
// its exponent bytes, so grouping like bytes together compresses far better
// than interleaved IEEE-754 does. It is increasingly the default for
// FLOAT/DOUBLE, which is why a reader without it cannot open modern numeric
// datasets.
//
// Transpose into PLAIN layout and hand it to decode_plain_values. Doing the
// conversion inline instead would mean restating the f32->f64 widening, the
// integer widen/sign-extend rules and the FLBA path -- four places to drift
// from the PLAIN decoder. A page is bounded, the scratch is one arena bump,
// and the transpose is the only new logic.
bool decode_byte_stream_split(ColCtx* cx, const uint8_t* v, uint64_t vlen,
                              const uint32_t* def, int64_t row0, uint32_t nvals,
                              uint32_t nvalid, Arena* arena) noexcept {
    assert(cx != nullptr && arena != nullptr);
    // The width is the PHYSICAL one, from the schema -- not cx->src_width.
    // src_width is a decode control that only the paths needing a width
    // conversion set: for FLOAT it stays at its default of cx->elem (8, the
    // Float64 output stride) because plain_f32_widen hardcodes its 4-byte
    // source instead of consulting it. Using it here made the f32 case demand
    // 8 bytes per value from a 4-byte-per-value page and fail closed -- caught
    // by the per-type fixtures, and invisible in a file whose float column
    // happened to be DOUBLE.
    uint32_t w = 0;
    switch (cx->pc->physical) {
        case PqType::Float:  case PqType::Int32:  w = 4u; break;
        case PqType::Double: case PqType::Int64:  w = 8u; break;
        case PqType::FixedLenByteArray:
            w = static_cast<uint32_t>(cx->pc->type_length);
            break;
        default: return false;      // BOOLEAN / BYTE_ARRAY: not a fixed width
    }
    // The spec allows FLOAT/DOUBLE and, since 2.9, the fixed-width integer and
    // FLBA types. Anything whose element width we do not know fails closed.
    if (w == 0u || w > 16u) return false;
    if (nvalid == 0u) {
        return decode_plain_values(cx, v, 0, def, row0, nvals, 0);
    }
    const uint64_t n = nvalid;
    if (n * w > vlen) return false;              // page shorter than it claims
    auto* plain = static_cast<uint8_t*>(arena->allocate(n * w, 8));
    if (plain == nullptr) return false;
    // Stream-major outer loop: each pass reads one stream sequentially and
    // writes with stride w. The opposite nesting would read w streams
    // scattered per value.
    for (uint32_t j = 0; j < w; ++j) {
        const uint8_t* BOLT_RESTRICT s = v + static_cast<uint64_t>(j) * n;
        uint8_t* BOLT_RESTRICT d = plain + j;
        for (uint64_t i = 0; i < n; ++i) d[i * w] = s[i];
    }
    return decode_plain_values(cx, plain, n * w, def, row0, nvals, nvalid);
}

// DATA_PAGE (v1): [def levels if OPTIONAL: u32 LE byte-len + RLE hybrid]
// then PLAIN values or [bit-width byte + RLE hybrid dictionary indices].
bool decode_values_by_encoding(ColCtx* cx, const uint8_t* v, uint64_t vlen,
                               const PageHdr* h, const uint32_t* def,
                               int64_t row0, uint32_t nvals, uint32_t nvalid,
                               Arena* arena) noexcept;

bool decode_data_page(ColCtx* cx, const uint8_t* page, uint64_t plen,
                      const PageHdr* h, int64_t row0, int64_t rows_left,
                      Arena* arena) noexcept {
    assert(cx != nullptr && h != nullptr);
    assert(arena != nullptr);
    if (h->nvals <= 0 || h->nvals > rows_left) return false;
    const uint32_t nvals = static_cast<uint32_t>(h->nvals);
    const uint8_t* v = page;
    uint64_t vlen = plen;
    uint32_t* def = nullptr;
    uint32_t nvalid = nvals;
    if (cx->pc->optional != 0) {                    // max_def_level == 1
        if (h->def_enc != kEncRle) return false;
        uint32_t dl = 0;
        uint32_t skip = 0;
        if (h->v2 != 0) {
            // v2 carries the length in the header and stores repetition levels
            // first; neither section has v1's 4-byte prefix.
            dl = static_cast<uint32_t>(h->def_len);
            skip = static_cast<uint32_t>(h->rep_len);
            if (static_cast<uint64_t>(skip) + dl > vlen) return false;
            v += skip;
            vlen -= skip;
        } else {
            if (vlen < 4) return false;
            std::memcpy(&dl, v, 4);
            skip = 4u;
            if (4ull + dl > vlen) return false;
            v += 4u;
            vlen -= 4u;
        }
        // Nullable but null-free is the common case (see rle_def_all_present).
        // Settle it from the run headers instead of expanding the levels only
        // to sum them and throw them away: same `def == nullptr` dense path,
        // without the allocation, the fill, or the sum.
        const uint32_t max_def = cx->pc->max_def;
        const uint32_t lbw = level_bit_width(max_def);
        if (rle_def_all_present(v, dl, nvals, max_def, lbw)) {
            def = nullptr;
            nvalid = nvals;
            v += dl;
            vlen -= dl;
        } else {
            def = static_cast<uint32_t*>(
                arena->allocate(uint64_t{nvals} * 4u, 4));
            if (def == nullptr) return false;
            if (!rle_hybrid_decode(v, dl, lbw, nvals, def)) return false;
            v += dl;
            vlen -= dl;
            nvalid = 0;
            for (uint32_t i = 0; i < nvals; ++i) {
                nvalid += (def[i] == max_def) ? 1u : 0u;
            }
            if (nvalid == nvals) def = nullptr;     // all-valid: dense path
            if (def != nullptr && cx->validity == nullptr) {
                return false;                       // chunk stats said no nulls
            }
        }
    }
    return decode_values_by_encoding(cx, v, vlen, h, def, row0, nvals, nvalid,
                                     arena);
}

// Value decode for a page whose LEVELS have already been consumed. Split out
// of decode_data_page so the LIST path can reuse every encoding: a list page
// carries the same values behind a different level layout, and duplicating
// this dispatch is how the two would silently drift on the next encoding.
//
// `def` is either null (dense: nvals values, all valid) or one level per
// OUTPUT slot; row0 is the first output slot.
bool decode_values_by_encoding(ColCtx* cx, const uint8_t* v, uint64_t vlen,
                               const PageHdr* h, const uint32_t* def,
                               int64_t row0, uint32_t nvals, uint32_t nvalid,
                               Arena* arena) noexcept {
    assert(cx != nullptr && h != nullptr && arena != nullptr);
    assert(nvalid <= nvals);
    if (h->enc == kEncPlain) {
        cx->codes_ok = 0;   // PLAIN page: no dictionary codes
        return decode_plain_values(cx, v, vlen, def, row0, nvals, nvalid);
    }
    if (h->enc == kEncDeltaBinaryPacked) {
        cx->codes_ok = 0;                 // not dictionary-coded
        return delta_binary_packed(cx, v, vlen, def, row0, nvals, nvalid, arena);
    }
    if (h->enc == kEncDeltaLenByteArray) {
        cx->codes_ok = 0;                 // not dictionary-coded
        return delta_length_byte_array(cx, v, vlen, def, row0, nvals, nvalid,
                                       arena);
    }
    if (h->enc == kEncDeltaByteArray) {
        cx->codes_ok = 0;                 // not dictionary-coded
        return delta_byte_array(cx, v, vlen, def, row0, nvals, nvalid, arena);
    }
    if (h->enc == kEncByteStreamSplit) {
        cx->codes_ok = 0;                 // not dictionary-coded
        return decode_byte_stream_split(cx, v, vlen, def, row0, nvals, nvalid,
                                        arena);
    }
    if (h->enc == kEncRle && cx->pc->physical == PqType::Boolean) {
        cx->codes_ok = 0;                 // not dictionary-coded
        return rle_bool(cx, v, vlen, def, row0, nvals, nvalid, arena);
    }
    if (h->enc == kEncPlainDict || h->enc == kEncRleDict) {
        if (nvalid == 0) return true;               // all-null page
        if (vlen < 1) return false;
        const uint32_t bw = v[0];
        if (bw > 32u) return false;
        uint32_t* idx = static_cast<uint32_t*>(
            arena->allocate(uint64_t{nvalid} * 4u, 4));
        if (idx == nullptr) return false;
        if (!rle_hybrid_decode(v + 1, vlen - 1, bw, nvalid, idx)) {
            return false;
        }
        return dict_gather(cx, idx, def, row0, nvals, nvalid);
    }
    return false;                                   // encoding outside v1
}

// Decompress one page payload per the chunk's codec, into arena memory.
//
// `zscratch` is a lazily-allocated scratch block reused across every zstd page
// in the chunk: it is ~150 KB, far too big to allocate per page and far too big
// for a stack frame. It stays null for chunks that never hit a zstd page.
bool decompress_page(PqCodec codec, const uint8_t* src, uint64_t src_len,
                     int32_t unc, Arena* arena, void** zscratch,
                     const uint8_t** out, uint64_t* out_len) noexcept {
    assert(arena != nullptr && zscratch != nullptr);
    assert(out != nullptr && out_len != nullptr);
    if (unc < 0) return false;
    const uint64_t unc_len = static_cast<uint64_t>(unc);
    if (codec == PqCodec::Uncompressed) {
        if (src_len != unc_len) return false;
        *out = src; *out_len = src_len;
        return true;
    }
    uint8_t* dst = static_cast<uint8_t*>(arena->allocate(unc_len + 1u, 8));
    if (dst == nullptr) return false;
    if (codec == PqCodec::Snappy) {
        if (!snappy_decompress(src, src_len, dst, unc_len)) return false;
    } else if (codec == PqCodec::Gzip) {
        // Parquet's GZIP is the gzip CONTAINER, not a bare deflate stream, and
        // gzip_decompress inflates raw deflate -- so the frame has to come off
        // first. Header is 10 bytes plus whatever the flag bits add; trailer is
        // CRC32 + ISIZE. Everything is bounds-checked before it is skipped: a
        // truncated frame must fail, not read past the page.
        uint64_t off = 10u;
        if (src_len < 18u) return false;                 // 10 header + 8 trailer
        if (src[0] != 0x1Fu || src[1] != 0x8Bu || src[2] != 0x08u) return false;
        const uint8_t flg = src[3];
        if ((flg & 0x04u) != 0u) {                       // FEXTRA
            if (off + 2u > src_len) return false;
            const uint64_t xlen = static_cast<uint64_t>(src[off]) |
                                  (static_cast<uint64_t>(src[off + 1]) << 8);
            off += 2u + xlen;
        }
        if ((flg & 0x08u) != 0u) {                       // FNAME, NUL-terminated
            while (off < src_len && src[off] != 0u) ++off;
            ++off;
        }
        if ((flg & 0x10u) != 0u) {                       // FCOMMENT
            while (off < src_len && src[off] != 0u) ++off;
            ++off;
        }
        if ((flg & 0x02u) != 0u) off += 2u;              // FHCRC
        if (off + 8u > src_len) return false;
        // inflate_raw, NOT gzip_decompress: the latter wraps zlib behind an
        // optional find_package, and this file's own bolt_inflate.h states the
        // rule -- "a reader that needs a find_package to open a real table is
        // not a reader". inflate_raw is bolt's self-contained RFC 1951 decoder.
        uint64_t got = 0;
        if (inflate_raw(src + off, src_len - off - 8u, dst, unc_len, &got)
                != kInflateOk) {
            return false;
        }
        if (got != unc_len) return false;        // page header size must match
    } else if (codec == PqCodec::Lz4Raw) {
        // LZ4_RAW is a bare LZ4 block. The legacy LZ4 codec (5) is the
        // Hadoop-framed variant and is deliberately NOT accepted here: it would
        // need the frame stripped too, and no writer in this tree emits it.
        // bolt's OWN block decoder, not bolt_lz4.h's liblz4 shim: that shim
        // is behind find_package(lz4) and absent from a default build, so an
        // LZ4_RAW file simply did not open. Same reasoning as inflate_raw
        // above and the self-contained zstd decoder -- a reader that needs a
        // find_package to open a real table is not a reader.
        if (!lz4_raw_decompress(src, src_len, dst, unc_len)) return false;
    } else if (codec == PqCodec::Zstd) {
        if (*zscratch == nullptr) {
            *zscratch = arena->allocate(zstd_scratch_size(), 8);
            if (*zscratch == nullptr) return false;
        }
        uint64_t got = 0;
        if (zstd_decode_raw(src, src_len, dst, unc_len, &got,
                            *zscratch, zstd_scratch_size()) != kZstdOk) {
            return false;
        }
        if (got != unc_len) return false;      // page header size must match
    } else {
        return false;                          // codec outside v1
    }
    *out = dst; *out_len = unc_len;
    return true;
}

bool assemble_v2_page(PqCodec codec, const uint8_t* src, uint64_t src_len,
                      const PageHdr* h, Arena* arena, void** zscratch,
                      const uint8_t** out, uint64_t* out_len) noexcept;


// One column chunk: walk its pages, decode rows [row0, row0+rows).
bool decode_chunk(const uint8_t* buf, uint64_t len, const PqChunk* ch,
                  ColCtx* cx, int64_t row0, int64_t rows,
                  Arena* arena) noexcept {
    assert(buf != nullptr && ch != nullptr);
    assert(cx != nullptr && arena != nullptr);
    if (rows == 0) return true;
    const int64_t start = (ch->dictionary_page_offset > 0)
        ? ch->dictionary_page_offset : ch->data_page_offset;
    if (start <= 0 || ch->total_compressed_size <= 0) return false;
    const uint64_t s = static_cast<uint64_t>(start);
    const uint64_t region = static_cast<uint64_t>(ch->total_compressed_size);
    if (s > len || region > len - s) return false;
    const uint64_t end = s + region;
    uint64_t p = s;
    int64_t rows_done = 0;
    void* zscratch = nullptr;                 // lazily allocated, chunk-scoped
    cx->dict = nullptr;                       // dictionary is chunk-scoped
    cx->dict_n = 0;
    for (uint32_t page = 0; page < kPqMaxPagesPerChunk; ++page) {  // bounded
        if (rows_done >= rows) break;
        if (p >= end) return false;           // ran out of pages
        TcCursor c{buf + p, buf + end};
        PageHdr h;
        if (!parse_page_header(&c, &h)) return false;
        const uint64_t pay = p + static_cast<uint64_t>(c.p - (buf + p));
        if (static_cast<uint64_t>(h.cmp) > end - pay) return false;
        const uint8_t* pd = buf + pay;
        uint64_t pd_len = static_cast<uint64_t>(h.cmp);
        if (h.type == kPageDataV2) {
            if (!assemble_v2_page(ch->codec, pd, pd_len, &h, arena, &zscratch,
                                  &pd, &pd_len)) {
                return false;
            }
        } else if (!decompress_page(ch->codec, pd, pd_len, h.unc, arena,
                                    &zscratch, &pd, &pd_len)) {
            return false;
        }
        if (h.type == kPageDict) {
            if (!decode_dict_page(cx, pd, pd_len, h.nvals, arena)) {
                return false;
            }
        } else if (h.type == kPageData || h.type == kPageDataV2) {
            if (!decode_data_page(cx, pd, pd_len, &h, row0 + rows_done,
                                  rows - rows_done, arena)) {
                return false;
            }
            rows_done += h.nvals;
        } else if (h.type != kPageIndex) {
            return false;                     // unknown page type
        }
        p = pay + static_cast<uint64_t>(h.cmp);
    }
    // Timestamp/Time unit normalization to microseconds (G2FEAT-46), over
    // the chunk's output range [row0, row0+rows) — plain and dict alike.
    // Invalid rows keep garbage but their validity bit is already cleared.
    if (cx->ts_rescale != 0 && rows_done == rows) {
        int64_t* d = reinterpret_cast<int64_t*>(cx->out) + row0;
        for (int64_t i = 0; i < rows; ++i) {       // bounded: chunk rows
            if (cx->ts_rescale == 1) d[i] *= 1000;   // millis -> us
            else                     d[i] /= 1000;   // nanos  -> us
        }
    }
    return rows_done == rows;
}

// G2FEAT-152: publish the dictionary hint on a fully-decoded Flat column.
//
// The column KEEPS format == Flat and its gathered StringViews, so every
// existing consumer is unaffected; `dict_child` is only read when
// format == Dictionary today, which makes this purely additive. Shape:
//     col.dict_child            -> Flat Int32 column, data = per-row codes
//     col.dict_child->dict_child-> Flat Utf8 column, data = dict entries
// A consumer may hash each dictionary entry ONCE and then index by code
// instead of re-hashing the row's content. Correctness must never depend on
// the hint: it is absent whenever any page was not dictionary-encoded, and
// BoltColumn::clone_into drops it (a clone into another arena must not carry
// a pointer into the source arena).
bool attach_dict_hint(const ColCtx* cx, BoltColumn* col, Arena* arena) noexcept {
    assert(cx != nullptr && col != nullptr && arena != nullptr);
    if (cx->codes_ok == 0 || cx->codes == nullptr) return true;   // no hint
    if (cx->dict == nullptr || cx->dict_n == 0) return true;
    if (col->format != ColumnFormat::Flat || col->length <= 0) return true;
    auto* vals = arena->allocate_array<BoltColumn>(1);
    auto* keys = arena->allocate_array<BoltColumn>(1);
    if (vals == nullptr || keys == nullptr) return true;           // skip hint
    *vals = BoltColumn::make_empty();
    vals->format            = ColumnFormat::Flat;
    vals->type              = col->type;
    vals->type_size_bytes   = col->type_size_bytes;
    vals->data              = cx->dict;
    vals->length            = static_cast<int64_t>(cx->dict_n);
    vals->str_overflow_base = col->str_overflow_base;
    vals->arena             = arena;
    *keys = BoltColumn::make_empty();
    keys->format          = ColumnFormat::Flat;
    keys->type            = BoltType::Int32;
    keys->type_size_bytes = 4;
    keys->data            = cx->codes;
    keys->length          = col->length;
    keys->dict_child      = vals;
    keys->arena           = arena;
    col->dict_child = keys;
    return true;
}


bool init_col_ctx_any(const PqMeta* m, uint32_t c, uint32_t g0, uint32_t g1,
                      int64_t rows, Arena* arena, BoltColumn* col,
                      ColCtx* cx) noexcept;

// Flat-path context builder: one value per row, so a REPEATED leaf is refused
// here. Refusing THIS column rather than the file is deliberate -- a table with
// three scalar columns and one list field must still open, and a projection
// that omits the list must still read.
bool init_col_ctx(const PqMeta* m, uint32_t c, uint32_t g0, uint32_t g1,
                  int64_t rows, Arena* arena, BoltColumn* col,
                  ColCtx* cx) noexcept {
    assert(m != nullptr && cx != nullptr);
    assert(c < m->n_columns);
    if (m->columns[c].max_rep != 0u) {
        if (bolt_pq_diag()) {
            std::fprintf(stderr, "bolt parquet: column '%s' is REPEATED "
                         "(list/map, max_rep=%u) -- use "
                         "parquet_read_list_column\n",
                         m->columns[c].name,
                         static_cast<unsigned>(m->columns[c].max_rep));
        }
        return false;
    }
    return init_col_ctx_any(m, c, g0, g1, rows, arena, col, cx);
}

// Assemble a DATA_PAGE_V2 into one contiguous buffer shaped like the rest of
// the reader expects: [rep levels][def levels][values].
//
// v2 stores the levels UNCOMPRESSED ahead of the values and compresses only the
// values, so the page cannot be inflated as a single blob. Copy the levels
// through verbatim and decompress the remainder after them.
bool assemble_v2_page(PqCodec codec, const uint8_t* src, uint64_t src_len,
                      const PageHdr* h, Arena* arena, void** zscratch,
                      const uint8_t** out, uint64_t* out_len) noexcept {
    assert(h != nullptr && arena != nullptr && out != nullptr);
    const uint64_t lv = static_cast<uint64_t>(h->rep_len) +
                        static_cast<uint64_t>(h->def_len);
    if (lv > src_len) return false;
    if (static_cast<uint64_t>(h->unc) < lv) return false;
    const uint64_t val_unc = static_cast<uint64_t>(h->unc) - lv;
    if (h->compressed == 0u || codec == PqCodec::Uncompressed) {
        // Nothing to do: the page is already in the target shape.
        *out = src;
        *out_len = src_len;
        return true;
    }
    const uint8_t* vals = nullptr;
    uint64_t vals_len = 0;
    if (!decompress_page(codec, src + lv, src_len - lv,
                         static_cast<int32_t>(val_unc), arena, zscratch,
                         &vals, &vals_len)) {
        return false;
    }
    auto* whole = static_cast<uint8_t*>(arena->allocate(lv + vals_len, 8));
    if (whole == nullptr) return false;
    if (lv != 0u) std::memcpy(whole, src, lv);
    if (vals_len != 0u) std::memcpy(whole + lv, vals, vals_len);
    *out = whole;
    *out_len = lv + vals_len;
    return true;
}

// Allocate one output column + its decode context for chunk range
// [g0, g1) of column c. `rows` = total rows the column will hold.
// The context builder, with no opinion about repetition. init_col_ctx below
// keeps the flat path's refusal; the LIST path calls this directly, because
// there the variable value count per row is the point rather than the problem.
bool init_col_ctx_any(const PqMeta* m, uint32_t c, uint32_t g0, uint32_t g1,
                      int64_t rows, Arena* arena, BoltColumn* col,
                      ColCtx* cx) noexcept {
    assert(m != nullptr && cx != nullptr);
    assert(c < m->n_columns && col != nullptr);
    const PqColumn* pc = &m->columns[c];
    BoltType t;
    uint8_t scale = 0;
    if (!parquet_map_type(pc, &t, &scale)) return false;
    *col = BoltColumn::make_flat_alloc(rows, t, arena);
    col->logical = parquet_map_logical(pc);
    if (col->data == nullptr && rows > 0) return false;
    if (rows == 0) {       // typed empty column (make_flat_alloc bails at 0)
        *col = BoltColumn::make_empty();
        col->format = ColumnFormat::Flat;
        col->type = t;
        col->type_size_bytes = static_cast<uint16_t>(bolt::type_size(t));
    }
    col->decimal_scale = scale;
    bool needs_validity = false;
    uint64_t ovf = 0;
    for (uint32_t g = g0; g < g1; ++g) {       // bounded: n_row_groups
        const PqChunk* ch = &m->chunks[m->row_groups[g].chunk_off + c];
        if (pc->optional != 0 && ch->null_count != 0) needs_validity = true;
        if (pc->physical == PqType::ByteArray &&
            ch->total_uncompressed_size > 0) {
            ovf += static_cast<uint64_t>(ch->total_uncompressed_size);
        }
    }
    if (needs_validity && rows > 0) {
        const uint64_t nb = (static_cast<uint64_t>(rows) + 7u) / 8u;
        uint8_t* bm = static_cast<uint8_t*>(arena->allocate(nb, 1));
        if (bm == nullptr) return false;
        std::memset(bm, 0xFF, nb);             // pages clear the null bits
        col->validity = bm;
        col->stats.all_valid = false;
    }
    std::memset(cx, 0, sizeof(*cx));
    cx->pc = pc;
    cx->type = t;
    // G2FEAT-152: track per-row dictionary codes for Utf8 only -- that is where
    // re-hashing wide content per row actually costs (ClickBench `url` averages
    // 88 bytes over 100M rows), and it bounds the extra memory to 4 B/row on one
    // column rather than every column. MUST come after the memset above, which
    // zeroes the whole context.
    if (t == BoltType::Utf8 && rows > 0) {
        auto* cb = static_cast<int32_t*>(
            arena->allocate(static_cast<uint64_t>(rows) * 4u, 4));
        if (cb != nullptr) {               // best-effort: no buffer => no hint
            cx->codes = cb;
            cx->codes_ok = 1;
        }
    }
    cx->elem = static_cast<uint32_t>(bolt::type_size(t));
    cx->out = static_cast<uint8_t*>(col->data);
    cx->validity = col->validity;
    // Type-conformance decode controls (G2FEAT-46). Default: source width ==
    // output width, signed. Refine per physical type + resolved logical.
    cx->src_width = cx->elem;
    cx->src_signed = 1;
    const bool out_unsigned = (t == BoltType::UInt8 || t == BoltType::UInt16 ||
                               t == BoltType::UInt32 || t == BoltType::UInt64);
    // Extension policy follows the SOURCE's signedness, not the output lane's
    // (G2FEAT-111). u32 now widens to the signed Int64 lane, so keying off the
    // output alone would sign-extend it and hand back the very negatives the
    // widening exists to avoid (4294967295 -> -1). A source annotated
    // INT(N, false) is zero-extended whatever lane it lands on; DECIMAL
    // mantissas and signed ints stay sign-extended.
    const bool src_unsigned_int =
        (pc->logical == static_cast<int32_t>(PqLogical::Int)) &&
        pc->int_signed == 0;
    const uint8_t ext_signed =
        (out_unsigned || src_unsigned_int) ? uint8_t{0} : uint8_t{1};
    if (pc->physical == PqType::Int96) {
        cx->is_int96 = 1;                 // 12-byte legacy timestamp -> us
        cx->src_width = 12;
    } else if (pc->physical == PqType::Int32) {
        cx->src_width = 4;
        cx->src_signed = ext_signed;      // Decimal64 mantissa stays signed
    } else if (pc->physical == PqType::Int64) {
        cx->src_width = 8;
        cx->src_signed = ext_signed;
    }
    // Integer width conversion needed only when source != output width and
    // the physical is an integer (Float widen / Bool / FLBA have own paths).
    cx->int_convert = ((pc->physical == PqType::Int32 ||
                        pc->physical == PqType::Int64) &&
                       cx->src_width != cx->elem) ? 1 : 0;
    // Timestamp/Duration output carries microseconds; rescale non-micro units.
    if (t == BoltType::Timestamp || t == BoltType::Duration) {
        if (pc->time_unit == 1) cx->ts_rescale = 1;        // millis -> us
        else if (pc->time_unit == 3) cx->ts_rescale = 2;   // nanos  -> us
    }
    cx->arena = arena;
    cx->ovf_base = &col->str_overflow_base;
    if (t == BoltType::Utf8 && ovf > 0 && rows > 0) {
        char* ob = static_cast<char*>(arena->allocate(ovf, 1));
        if (ob == nullptr) return false;
        cx->overflow = ob;
        cx->overflow_cap = ovf;
        col->str_overflow_base = ob;
    }
    return true;
}

}  // namespace

// ---- LIST columns (Dremel record assembly) --------------------------------
//
// A leaf under a REPEATED group produces a VARIABLE number of values per row,
// which is why every other path in this reader refuses it. Assembling it needs
// the repetition levels, which say where each row starts, and the definition
// levels, which say whether a slot is an element, a null element, an empty
// list or a null list. The four cases are decided by two levels recorded at
// schema-walk time (see PqColumn::list_def / rep_def):
//
//   def <  list_def            the LIST is NULL
//   list_def <= def < rep_def  the list is PRESENT but EMPTY
//   def >= rep_def             one element, NULL iff def < max_def
//
// An empty list and a null list are DIFFERENT VALUES. A null bitmap alone
// cannot distinguish them, which is why the offsets carry emptiness and the
// bitmap carries nullness, and why both levels are recorded rather than one
// being derived from max_def.
//
// Scope: max_rep == 1, a single level of repetition -- `list<T>`, and the two
// leaves of a `map<K,V>`. A list of lists (max_rep >= 2) is refused rather
// than guessed at, exactly as before.

// Per-page level decode for a repeated leaf. Fills `rep` and `def` with one
// entry per leaf slot and returns the values pointer/length that follow.
bool list_page_levels(const ColCtx* cx, const uint8_t* page, uint64_t plen,
                      const PageHdr* h, uint32_t nvals, uint32_t* rep,
                      uint32_t* def, const uint8_t** out_v,
                      uint64_t* out_vlen) noexcept {
    assert(cx != nullptr && h != nullptr);
    assert(rep != nullptr && def != nullptr);
    const uint8_t* v = page;
    uint64_t vlen = plen;
    const uint32_t max_rep = cx->pc->max_rep;
    const uint32_t max_def = cx->pc->max_def;
    // v1 prefixes each level stream with its own u32 byte length and stores
    // repetition FIRST; v2 stores both raw with the lengths in the header.
    // The flat path never had to read the repetition stream at all -- it only
    // ever skipped it -- so this is the one place that consumes it.
    uint32_t rl = 0, dl = 0;
    if (h->v2 != 0) {
        rl = static_cast<uint32_t>(h->rep_len);
        dl = static_cast<uint32_t>(h->def_len);
        if (static_cast<uint64_t>(rl) + dl > vlen) return false;
    } else {
        if (vlen < 4) return false;
        std::memcpy(&rl, v, 4);
        v += 4; vlen -= 4;
        if (rl > vlen) return false;
    }
    if (!rle_hybrid_decode(v, rl, level_bit_width(max_rep), nvals, rep)) {
        return false;
    }
    v += rl; vlen -= rl;
    if (h->v2 == 0) {
        if (vlen < 4) return false;
        std::memcpy(&dl, v, 4);
        v += 4; vlen -= 4;
        if (dl > vlen) return false;
    }
    if (!rle_hybrid_decode(v, dl, level_bit_width(max_def), nvals, def)) {
        return false;
    }
    v += dl; vlen -= dl;
    *out_v = v;
    *out_vlen = vlen;
    return true;
}

// Accumulated state across a chunk's pages.
struct ListBuild {
    uint32_t* rep;          // one per leaf slot, whole chunk
    uint32_t* def;
    int64_t   n_slots;      // slots filled so far
    int64_t   cap_slots;
};

// Decode one data page of a repeated leaf: record its levels, then decode the
// ELEMENT values densely into `cx` starting at element slot `elem0`. Returns
// the number of element slots this page contributed (elements, not rows).
bool list_decode_page(ColCtx* cx, const uint8_t* page, uint64_t plen,
                      const PageHdr* h, ListBuild* b, int64_t elem0,
                      uint32_t* out_elems, Arena* arena) noexcept {
    assert(cx != nullptr && b != nullptr && out_elems != nullptr);
    if (h->nvals <= 0) { *out_elems = 0; return true; }
    const uint32_t nvals = static_cast<uint32_t>(h->nvals);
    if (b->n_slots + nvals > b->cap_slots) return false;
    uint32_t* rep = b->rep + b->n_slots;
    uint32_t* def = b->def + b->n_slots;
    const uint8_t* v = nullptr;
    uint64_t vlen = 0;
    if (!list_page_levels(cx, page, plen, h, nvals, rep, def, &v, &vlen)) {
        return false;
    }
    b->n_slots += nvals;

    // Compact the slots that actually carry an element. The empty/null-list
    // markers occupy a leaf slot but no element, so passing the raw def array
    // straight through would reserve an element slot for them and leave a gap
    // no list's offset range points at.
    const uint32_t rep_def = cx->pc->rep_def;
    const uint32_t max_def = cx->pc->max_def;
    uint32_t n_elem = 0, nvalid = 0;
    for (uint32_t i = 0; i < nvals; ++i) {
        const bool has_elem = def[i] >= rep_def;
        n_elem += has_elem ? 1u : 0u;
        nvalid += (def[i] == max_def) ? 1u : 0u;
    }
    *out_elems = n_elem;
    if (n_elem == 0) return true;                 // page is all empty lists
    uint32_t* edef = static_cast<uint32_t*>(
        arena->allocate(uint64_t{n_elem} * 4u, 4));
    if (edef == nullptr) return false;
    uint32_t k = 0;
    for (uint32_t i = 0; i < nvals; ++i) {
        if (def[i] >= rep_def) edef[k++] = def[i];
    }
    assert(k == n_elem);
    // A page whose elements are all non-null takes the dense path, which is
    // both faster and what every non-nullable list hits.
    const uint32_t* pass = (nvalid == n_elem) ? nullptr : edef;
    return decode_values_by_encoding(cx, v, vlen, h, pass, elem0, n_elem,
                                     nvalid, arena);
}

// Walk a repeated leaf's chunk and assemble the LIST column.
// Assemble a repeated leaf across the row-group range [g0, g1) into one list
// column. A range rather than a single group because parquet_read_file builds
// ONE batch for the whole file, exactly as the flat path does with
// init_col_ctx_any(meta, c, 0, n_row_groups, ...).
//
// Row groups are independent on the wire -- each has its own chunk region and
// its own dictionary -- so the walk resets the dictionary per group. Getting
// that wrong is a silent misread, not a crash: group 1 would gather through
// group 0's dictionary and return real strings from the wrong rows.
bool build_list_column(const uint8_t* buf, uint64_t len, const PqMeta* meta,
                       uint32_t g0, uint32_t g1, uint32_t col, Arena* arena,
                       BoltColumn* out_col, int64_t* out_rows) noexcept {
    assert(buf != nullptr && meta != nullptr && arena != nullptr);
    assert(out_col != nullptr && out_rows != nullptr);
    if (g0 >= g1 || g1 > meta->n_row_groups) return false;
    const PqColumn* pc = &meta->columns[col];
    int64_t n_slots = 0, n_rows_total = 0;
    for (uint32_t g = g0; g < g1; ++g) {
        const PqChunk* c2 = &meta->chunks[meta->row_groups[g].chunk_off + col];
        if (c2->num_values < 0) return false;
        n_slots += c2->num_values;
        n_rows_total += meta->row_groups[g].num_rows;
    }
    *out_rows = 0;
    if (n_slots == 0) {
        // Chunks with no leaf values still have rows, every one of them an
        // empty list. Offsets are all zero and there is no element.
        BoltColumn elem = BoltColumn::make_flat_alloc(1, BoltType::Int64, arena);
        elem.length = 0;
        auto* offs = arena->allocate_array<int32_t>(n_rows_total + 1);
        if (offs == nullptr) return false;
        for (int64_t i = 0; i <= n_rows_total; ++i) offs[i] = 0;
        *out_col = BoltColumn::make_list(&elem, offs, n_rows_total, nullptr,
                                         arena);
        *out_rows = n_rows_total;
        return true;
    }

    // Element column, sized to the upper bound: every leaf slot yields at
    // most one element. The exact count is only known after the levels are
    // read, so the length is trimmed at the end rather than guessed at.
    BoltColumn elem = BoltColumn::make_empty();
    ColCtx cx;
    if (!init_col_ctx_any(meta, col, g0, g1, n_slots, arena, &elem, &cx)) {
        return false;
    }
    // Elements can be null whenever there is a definition level between the
    // repeated node and the leaf; init_col_ctx_any sized validity from the
    // chunk's null_count, which counts NULL ELEMENTS and so is the right test.
    int64_t range_nulls = 0;
    for (uint32_t g = g0; g < g1; ++g) {
        range_nulls += meta->chunks[meta->row_groups[g].chunk_off + col].null_count;
    }
    if (cx.validity == nullptr && pc->max_def > pc->rep_def &&
        range_nulls != 0) {
        const uint64_t nb = (static_cast<uint64_t>(n_slots) + 7u) / 8u;
        auto* bm = static_cast<uint8_t*>(arena->allocate(nb, 1));
        if (bm == nullptr) return false;
        std::memset(bm, 0xFF, nb);
        elem.validity = bm;
        elem.stats.all_valid = false;
        cx.validity = bm;
    }

    ListBuild b;
    b.rep = static_cast<uint32_t*>(arena->allocate(uint64_t(n_slots) * 4u, 4));
    b.def = static_cast<uint32_t*>(arena->allocate(uint64_t(n_slots) * 4u, 4));
    if (b.rep == nullptr || b.def == nullptr) return false;
    b.n_slots = 0;
    b.cap_slots = n_slots;

    int64_t elem_n = 0;
    void* zscratch = nullptr;
    for (uint32_t g = g0; g < g1; ++g) {
        const PqChunk* ch = &meta->chunks[meta->row_groups[g].chunk_off + col];
        const int64_t start = (ch->dictionary_page_offset > 0)
            ? ch->dictionary_page_offset : ch->data_page_offset;
        if (start <= 0 || ch->total_compressed_size <= 0) return false;
        const uint64_t s = static_cast<uint64_t>(start);
        const uint64_t region = static_cast<uint64_t>(ch->total_compressed_size);
        if (s > len || region > len - s) return false;
        const uint64_t end = s + region;
        uint64_t p = s;
        // PER CHUNK, not per column: a dictionary belongs to one column chunk,
        // so carrying group 0's dictionary into group 1 would gather real
        // values from the wrong rows -- a silent misread.
        cx.dict = nullptr;
        cx.dict_n = 0;
        const int64_t want = b.n_slots + ch->num_values;
        for (uint32_t page = 0; page < kPqMaxPagesPerChunk; ++page) {
            if (b.n_slots >= want) break;
            if (p >= end) return false;
            TcCursor c{buf + p, buf + end};
            PageHdr h;
            if (!parse_page_header(&c, &h)) return false;
            const uint64_t pay = p + static_cast<uint64_t>(c.p - (buf + p));
            if (static_cast<uint64_t>(h.cmp) > end - pay) return false;
            const uint8_t* pd = buf + pay;
            uint64_t pd_len = static_cast<uint64_t>(h.cmp);
            if (h.type == kPageDataV2) {
                if (!assemble_v2_page(ch->codec, pd, pd_len, &h, arena,
                                      &zscratch, &pd, &pd_len)) {
                    return false;
                }
            } else if (!decompress_page(ch->codec, pd, pd_len, h.unc, arena,
                                        &zscratch, &pd, &pd_len)) {
                return false;
            }
            if (h.type == kPageDict) {
                if (!decode_dict_page(&cx, pd, pd_len, h.nvals, arena)) {
                    return false;
                }
            } else if (h.type == kPageData || h.type == kPageDataV2) {
                uint32_t got = 0;
                if (!list_decode_page(&cx, pd, pd_len, &h, &b, elem_n, &got,
                                      arena)) {
                    return false;
                }
                elem_n += got;
            } else if (h.type != kPageIndex) {
                return false;
            }
            p = pay + static_cast<uint64_t>(h.cmp);
        }
        if (b.n_slots != want) return false;
    }
    if (b.n_slots != n_slots) return false;

    // ---- assembly: rep says where rows start, def says what each slot is ---
    int64_t rows = 0;
    for (int64_t j = 0; j < n_slots; ++j) rows += (b.rep[j] == 0u) ? 1 : 0;
    if (rows != n_rows_total) return false;      // levels disagree with the footer
    auto* offs = arena->allocate_array<int32_t>(rows + 1);
    if (offs == nullptr) return false;
    const uint64_t vb = (static_cast<uint64_t>(rows) + 7u) / 8u;
    auto* lval = static_cast<uint8_t*>(arena->allocate(vb, 1));
    if (lval == nullptr) return false;
    std::memset(lval, 0xFF, vb);
    const uint32_t list_def = pc->list_def;
    const uint32_t rep_def = pc->rep_def;
    int64_t row = -1;
    int64_t ec = 0;
    bool any_null_list = false;
    for (int64_t j = 0; j < n_slots; ++j) {
        if (b.rep[j] == 0u) {
            ++row;
            offs[row] = static_cast<int32_t>(ec);
            if (b.def[j] < list_def) {           // the LIST itself is null
                lval[row >> 3] = static_cast<uint8_t>(
                    lval[row >> 3] & ~(1u << (row & 7)));
                any_null_list = true;
            }
        }
        ec += (b.def[j] >= rep_def) ? 1 : 0;
    }
    if (row + 1 != rows) return false;
    offs[rows] = static_cast<int32_t>(ec);
    if (ec != elem_n) return false;              // assembly disagrees with decode
    elem.length = elem_n;

    *out_col = BoltColumn::make_list(&elem, offs, rows,
                                     any_null_list ? lval : nullptr, arena);
    if (out_col->data == nullptr) return false;
    *out_rows = rows;
    return true;
}

// ---- public API ---------------------------------------------------------------

BoltLogical parquet_map_logical(const PqColumn* col) noexcept {
    if (col == nullptr) return BoltLogical::None;
    switch (static_cast<PqLogical>(col->logical)) {
        case PqLogical::Json:    return BoltLogical::Json;
        case PqLogical::Bson:    return BoltLogical::Bson;
        case PqLogical::Variant: return BoltLogical::Variant;
        default:                 return BoltLogical::None;
    }
}

bool parquet_map_type(const PqColumn* col, BoltType* out_type,
                      uint8_t* out_scale) noexcept {
    assert(out_type != nullptr);
    assert(out_scale != nullptr);
    if (col == nullptr) return false;
    *out_scale = 0;
    // Resolve the logical intent from either encoding (G2FEAT-46): the
    // normalized PqColumn::logical (LogicalType or ConvertedType-derived)
    // plus the legacy converted flags kept as a belt-and-suspenders check.
    const bool is_dec  = (col->converted == kConvDecimal) ||
                         (col->logical == static_cast<int32_t>(PqLogical::Decimal));
    const bool is_date = (col->converted == kConvDate) ||
                         (col->logical == static_cast<int32_t>(PqLogical::Date));
    const bool is_ts   = (col->logical == static_cast<int32_t>(PqLogical::Timestamp));
    const bool is_time = (col->logical == static_cast<int32_t>(PqLogical::Time));
    const bool is_int  = (col->logical == static_cast<int32_t>(PqLogical::Int));
    switch (col->physical) {
        case PqType::Int64:
            if (is_dec) {                               // int64 mantissa as-is
                if (col->scale < 0 || col->scale > 18) return false;
                *out_type = BoltType::Decimal64;
                *out_scale = static_cast<uint8_t>(col->scale);
                return true;
            }
            if (is_ts)   { *out_type = BoltType::Timestamp; return true; }
            if (is_time) { *out_type = BoltType::Duration;  return true; }
            // INTEGER WIDTH: deliberately NOT narrowed to exact unsigned types.
            // G2FEAT-46 originally mapped unsigned IntType to UInt8/16/32/64.
            // That is more type-faithful, but it would change the BoltType of
            // real production columns (e.g. the TAQ rowgroup u16/u64 columns)
            // and downstream lane support for exact unsigned is thin -- 4
            // BoltType::UInt16/UInt32 references in all of chukonu/src, 14 in
            // marbledb/src. Flipping it needs a downstream kernel audit, not a
            // reader change. Until then keep the signed lane, which every
            // kernel handles, and which is what main already produced.
            // Tracked as a follow-up; see the note in tests/test_bolt_parquet_types.cpp.
            *out_type = BoltType::Int64;
            return true;
        case PqType::Int32:
            if (is_dec) {                               // INT32-backed DECIMAL
                if (col->scale < 0 || col->scale > 18) return false;
                *out_type = BoltType::Decimal64;        // p<=9 always fits
                *out_scale = static_cast<uint8_t>(col->scale);
                return true;
            }
            if (is_date) { *out_type = BoltType::Date32; return true; }
            if (is_time) { *out_type = BoltType::Int32;  return true; }  // ms raw
            // See the INTEGER WIDTH note on the Int64 branch above: exact
            // unsigned lanes are not emitted; every unsigned width instead
            // takes the NEXT WIDER SIGNED lane, which is the narrowest lane
            // that holds every value it can carry.
            if (is_int) {
                // SIGNED sub-word widths are honoured exactly.
                //
                // UNSIGNED takes the next wider SIGNED lane, per width -- NOT
                // one lane for all three (G2FEAT-111). u8 (0..255) and u16
                // (0..65535) fit Int32 losslessly. u32 does NOT: 2^31..2^32-1
                // is over half its range, and mapping it to Int32 made those
                // values read back NEGATIVE (4294967295 -> -1). The comment
                // that used to sit here claimed an INT32-physical unsigned
                // value "is always representable in Int32", which is false for
                // exactly that half of u32; it is corrected rather than
                // deleted because it is what stopped the previous reader from
                // checking. u32 therefore widens to Int64, and init_col_ctx
                // ZERO-extends it at decode (src_signed = 0) -- sign-extending
                // would reintroduce the same negatives one layer down.
                //
                // u64 has no wider signed lane and stays a raw Int64 bit
                // pattern above 2^63-1; fixing that needs a real UInt64 lane
                // plus the downstream kernel audit the Int64 note describes,
                // not a widening. golden_taq_types.parquet pins the current
                // bit-pattern behaviour.
                if (col->int_signed) {
                    switch (col->int_bits) {
                        case 8:  *out_type = BoltType::Int8;  return true;
                        case 16: *out_type = BoltType::Int16; return true;
                        case 32: *out_type = BoltType::Int32; return true;
                        default: break;  // 64-bit on INT32 physical: malformed
                    }
                } else {
                    switch (col->int_bits) {
                        // u8/u16 fit signed 32 -- unchanged, and what main
                        // already produced for the TAQ u16 column.
                        case 8: case 16:
                            *out_type = BoltType::Int32; return true;
                        case 32:
                            *out_type = BoltType::Int64; return true;
                        default: break;
                    }
                }
            }
            *out_type = BoltType::Int32;
            return true;
        case PqType::Int96:                   // legacy timestamp -> us at decode
            *out_type = BoltType::Timestamp;
            return true;
        case PqType::Double:
            *out_type = BoltType::Float64;
            return true;
        case PqType::Float:                   // f32 widened at decode
            *out_type = BoltType::Float64;
            return true;
        case PqType::Boolean:                 // 0/1
            *out_type = BoltType::Int64;
            return true;
        case PqType::ByteArray:
            *out_type = BoltType::Utf8;
            return true;
        case PqType::FixedLenByteArray:
            if (!is_dec) return false;         // non-DECIMAL FLBA: reject (v1)
            if (col->type_length <= 0 || col->type_length > 16) return false;
            if (col->scale < 0 || col->scale > 38) return false;
            if (col->precision <= 18) {                 // W-DEC representation
                *out_type = BoltType::Decimal64;
            } else {
                *out_type = BoltType::Decimal128;
            }
            *out_scale = static_cast<uint8_t>(col->scale);
            return true;
        default:
            return false;
    }
}

bool parquet_schema_from_meta(const PqMeta* meta, BoltSchema* out,
                              bool lowercase_names) noexcept {
    assert(meta != nullptr);
    assert(out != nullptr);
    if (meta->n_columns == 0u) return false;
    // Caller owns the storage (BoltSchema::set_storage) — we never allocate.
    if (out->fields == nullptr || out->cap < meta->n_columns) return false;

    out->num_fields = 0;
    for (uint32_t c = 0; c < meta->n_columns; ++c) {
        const PqColumn& pc = meta->columns[c];
        BoltType bt    = BoltType::NA;
        uint8_t  scale = 0;
        // Same mapping the decoder uses — the schema cannot drift from it.
        if (!parquet_map_type(&pc, &bt, &scale)) return false;

        BoltField& f = out->fields[c];
        std::memset(&f, 0, sizeof(f));
        size_t i = 0;
        for (; pc.name[i] != '\0' && i < kMaxFieldName; ++i) {
            const char ch = pc.name[i];
            f.name[i] = (lowercase_names && ch >= 'A' && ch <= 'Z')
                            ? static_cast<char>(ch + 32)
                            : ch;
        }
        f.name[i]       = '\0';
        f.type          = bt;
        f.nullable      = (pc.optional != 0u);
        f.decimal_scale = scale;   // 0 for non-decimal columns
        ++out->num_fields;
    }
    assert(out->num_fields == meta->n_columns);
    return true;
}

// Decode one raw parquet statistic blob into an int64 in catalog units.
// Mirrors parquet_map_type's own physical/logical dispatch so a stat can
// never be interpreted differently from the column it describes.
bool pq_stat_to_i64(const PqColumn& col, const uint8_t* b, uint32_t len,
                    int64_t* out) noexcept {
    assert(b != nullptr);
    assert(out != nullptr);
    if (len == 0u) return false;
    const bool is_dec = (col.converted == kConvDecimal) ||
                        (col.logical == static_cast<int32_t>(PqLogical::Decimal));
    switch (col.physical) {
        case PqType::Int32: {                 // INT32, DATE, INT32-DECIMAL
            if (len != 4u) return false;
            int32_t v = 0;
            std::memcpy(&v, b, 4);            // parquet stats are little-endian
            *out = static_cast<int64_t>(v);   // DATE lands as day number
            return true;
        }
        case PqType::Int64: {                 // INT64, INT64-DECIMAL mantissa
            if (len != 8u) return false;
            int64_t v = 0;
            std::memcpy(&v, b, 8);
            *out = v;
            return true;
        }
        case PqType::FixedLenByteArray: {     // DECIMAL(p<=18) as big-endian FLBA
            if (!is_dec || len == 0u || len > 8u) return false;
            // Sign-extend from the top byte, then accumulate big-endian.
            int64_t v = (b[0] & 0x80u) ? -1 : 0;
            for (uint32_t i = 0; i < len; ++i) {
                v = static_cast<int64_t>((static_cast<uint64_t>(v) << 8) | b[i]);
            }
            *out = v;
            return true;
        }
        default:
            // Utf8/Binary/float/double/bool/Int96: no meaningful int64 range.
            return false;
    }
}

bool parquet_column_int_range(const PqMeta* meta, uint32_t col_idx,
                              int64_t* out_min, int64_t* out_max) noexcept {
    assert(meta != nullptr);
    assert(out_min != nullptr && out_max != nullptr);
    if (col_idx >= meta->n_columns) return false;
    if (meta->n_row_groups == 0u || meta->chunks == nullptr) return false;

    const PqColumn& col = meta->columns[col_idx];
    int64_t lo = 0, hi = 0;
    bool seen = false;
    for (uint32_t g = 0; g < meta->n_row_groups; ++g) {
        const PqRowGroup& rg = meta->row_groups[g];
        if (col_idx >= rg.chunk_count) return false;          // ragged: unproven
        const PqChunk& ch = meta->chunks[rg.chunk_off + col_idx];
        // A chunk with no rows carries no range but does not invalidate one.
        if (ch.num_values == 0) continue;
        int64_t cmin = 0, cmax = 0;
        if (!pq_stat_to_i64(col, ch.min_bytes, ch.min_len, &cmin)) return false;
        if (!pq_stat_to_i64(col, ch.max_bytes, ch.max_len, &cmax)) return false;
        if (cmin > cmax) return false;                        // corrupt stat
        if (!seen) { lo = cmin; hi = cmax; seen = true; }
        else {
            if (cmin < lo) lo = cmin;
            if (cmax > hi) hi = cmax;
        }
    }
    if (!seen) return false;                                  // all chunks empty
    *out_min = lo;
    *out_max = hi;
    return true;
}

bool parquet_read_meta(const uint8_t* buf, uint64_t len, Arena* arena,
                       PqMeta* out) noexcept {
    assert(arena != nullptr);
    assert(out != nullptr);
    uint64_t off = 0;
    uint32_t mlen = 0;
    if (!pq_locate_footer(buf, len, &off, &mlen)) return false;
    // First pass with a modest chunk table; retry once at the hard cap
    // (chunk-heavy files). A corrupt footer re-fails identically.
    out->chunks = arena->allocate_array<PqChunk>(kPqMetaChunksFirst);
    out->chunks_cap = (out->chunks != nullptr) ? kPqMetaChunksFirst : 0;
    if (out->chunks == nullptr) return false;
    if (pq_parse_file_meta(buf + off, mlen, out)) return true;
    const uint32_t cap = kPqMaxRowGroups * kPqMaxColumns;
    out->chunks = arena->allocate_array<PqChunk>(cap);
    if (out->chunks == nullptr) return false;
    out->chunks_cap = cap;
    return pq_parse_file_meta(buf + off, mlen, out);
}

bool parquet_read_row_group(const uint8_t* buf, uint64_t len,
                            const PqMeta* meta, uint32_t row_group,
                            Arena* arena, BoltColumn* out_cols,
                            int64_t* out_rows) noexcept {
    assert(meta != nullptr && out_cols != nullptr);
    assert(out_rows != nullptr);
    if (buf == nullptr || arena == nullptr) return false;
    if (row_group >= meta->n_row_groups) return false;
    const PqRowGroup* rg = &meta->row_groups[row_group];
    if (rg->num_rows < 0) return false;
    for (uint32_t c = 0; c < meta->n_columns; ++c) {    // bounded: <= kPqMaxColumns (128)
        ColCtx cx;
        if (!init_col_ctx(meta, c, row_group, row_group + 1, rg->num_rows,
                          arena, &out_cols[c], &cx)) {
            return false;
        }
        const PqChunk* ch = &meta->chunks[rg->chunk_off + c];
        if (ch->num_values != rg->num_rows) return false;   // flat invariant
        if (!decode_chunk(buf, len, ch, &cx, 0, rg->num_rows, arena)) {
            return false;
        }
        (void)attach_dict_hint(&cx, &out_cols[c], arena);
    }
    *out_rows = rg->num_rows;
    return true;
}

bool parquet_read_row_group_cols(const uint8_t* buf, uint64_t len,
                                 const PqMeta* meta, uint32_t row_group,
                                 const uint16_t* col_idx, uint32_t n_idx,
                                 Arena* arena, BoltColumn* out_cols,
                                 int64_t* out_rows) noexcept {
    assert(meta != nullptr && out_cols != nullptr);
    assert(out_rows != nullptr);
    if (buf == nullptr || arena == nullptr) return false;
    if (col_idx == nullptr || n_idx == 0 || n_idx > meta->n_columns) return false;
    if (row_group >= meta->n_row_groups) return false;
    const PqRowGroup* rg = &meta->row_groups[row_group];
    if (rg->num_rows < 0) return false;
    for (uint32_t j = 0; j < n_idx; ++j) {              // bounded: <= n_columns
        const uint32_t c = col_idx[j];
        if (c >= meta->n_columns) return false;
        ColCtx cx;
        if (!init_col_ctx(meta, c, row_group, row_group + 1, rg->num_rows,
                          arena, &out_cols[j], &cx)) {
            return false;
        }
        const PqChunk* ch = &meta->chunks[rg->chunk_off + c];
        if (ch->num_values != rg->num_rows) return false;   // flat invariant
        if (!decode_chunk(buf, len, ch, &cx, 0, rg->num_rows, arena)) {
            return false;
        }
        (void)attach_dict_hint(&cx, &out_cols[j], arena);
    }
    *out_rows = rg->num_rows;
    return true;
}

// G2FEAT-49: resumable single-column page decode. Decodes WHOLE data pages of
// column `col` in `row_group`, starting at byte offset `start_off` (0 = first
// data page of the chunk), until >= `max_rows` rows are decoded (whole pages,
// may overshoot), into a FRESH Flat column `out_col` sized to exactly the
// decoded rows. Returns the decoded row count in *out_rows and the byte offset
// of the next undecoded page in *next_off (0 = chunk exhausted).
//
// v1 scope: PLAIN chunks only — fail-closed on a dictionary page (a chunk-scoped
// dictionary cannot be sub-decoded without re-decoding the whole dict per call,
// which defeats the footprint goal). Byte-exact at the VALUE level vs a
// whole-chunk decode of the same pages: whole pages, no mid-page trim, so
// concatenating sub-chunks reproduces the column (per-sub-chunk overflow means
// StringView offsets differ; the decoded string values are identical).
bool parquet_read_col_chunk_pages(const uint8_t* buf, uint64_t len,
                                  const PqMeta* meta, uint32_t row_group,
                                  uint16_t col, uint64_t start_off,
                                  int64_t max_rows, Arena* arena,
                                  BoltColumn* out_col, int64_t* out_rows,
                                  uint64_t* next_off) noexcept {
    assert(meta != nullptr && out_col != nullptr);
    assert(out_rows != nullptr && next_off != nullptr);
    if (buf == nullptr || arena == nullptr) return false;
    if (row_group >= meta->n_row_groups) return false;
    if (col >= meta->n_columns) return false;
    if (max_rows <= 0) return false;
    *out_rows = 0;
    *next_off = 0;
    const PqRowGroup* rg = &meta->row_groups[row_group];
    const PqChunk* ch = &meta->chunks[rg->chunk_off + col];
    if (ch->data_page_offset <= 0 || ch->total_compressed_size <= 0) return false;
    // A dictionary chunk's byte region starts at the DICTIONARY page --
    // total_compressed_size spans from there -- while the data pages this
    // function walks start at data_page_offset. Conflating the two truncates
    // the region by the dictionary page's length and the last data page then
    // reads as out of bounds.
    const uint64_t region_start = static_cast<uint64_t>(ch->data_page_offset);
    const uint64_t chunk_start = (ch->dictionary_page_offset > 0)
        ? static_cast<uint64_t>(ch->dictionary_page_offset) : region_start;
    if (chunk_start > region_start) return false;      // dict must precede data
    const uint64_t end =
        chunk_start + static_cast<uint64_t>(ch->total_compressed_size);
    if (end > len || end < region_start) return false;
    const uint64_t p_start = (start_off == 0) ? region_start : start_off;
    if (p_start < region_start || p_start > end) return false;
    if (p_start == end) return true;                    // exhausted (rows=0)
    // ---- Pass 1: header scan (no decompress) to size the sub-chunk ----
    uint64_t p = p_start;
    int64_t rows_sum = 0;
    uint64_t unc_sum = 0;
    uint64_t p_after = p_start;
    bool reached_end = false;
    for (uint32_t page = 0; page < kPqMaxPagesPerChunk; ++page) {
        if (rows_sum >= max_rows) break;
        if (p >= end) { reached_end = true; break; }
        TcCursor c{buf + p, buf + end};
        PageHdr h;
        if (!parse_page_header(&c, &h)) return false;
        const uint64_t pay = p + static_cast<uint64_t>(c.p - (buf + p));
        if (static_cast<uint64_t>(h.cmp) > end - pay) return false;
        const uint64_t p_next = pay + static_cast<uint64_t>(h.cmp);
        if (h.type == kPageData) {
            if (h.nvals < 0) return false;
            rows_sum += h.nvals;
            if (h.unc > 0) unc_sum += static_cast<uint64_t>(h.unc);
            p_after = p_next;
        } else if (h.type == kPageDict) {
            return false;                               // no dict after data
        } else if (h.type != kPageIndex) {
            return false;                               // DATA_PAGE_V2 etc.
        }
        p = p_next;
    }
    if (rows_sum == 0) return true;                      // no data pages left

    // For a dictionary column the STRING bytes live in the dictionary page;
    // the data pages hold only indices, so unc_sum above says nothing about
    // how much overflow room the Utf8 values need. Add the dictionary page's
    // uncompressed size. ensure_overflow_room grows on demand anyway, so an
    // under-estimate costs a realloc rather than a failure -- but starting
    // from the index-stream size alone would realloc on essentially every
    // dictionary Utf8 read.
    PageHdr dict_hdr;
    uint64_t dict_pay = 0;
    const bool has_dict = ch->dictionary_page_offset > 0;
    if (has_dict) {
        TcCursor dc{buf + chunk_start, buf + end};
        if (!parse_page_header(&dc, &dict_hdr)) return false;
        if (dict_hdr.type != kPageDict) return false;
        dict_pay = chunk_start +
                   static_cast<uint64_t>(dc.p - (buf + chunk_start));
        if (static_cast<uint64_t>(dict_hdr.cmp) > end - dict_pay) return false;
        if (dict_hdr.unc > 0) unc_sum += static_cast<uint64_t>(dict_hdr.unc);
    }
    // ---- Allocate a fresh sub-chunk column sized to rows_sum ----
    const PqColumn* pc = &meta->columns[col];
    BoltType t;
    uint8_t scale = 0;
    if (!parquet_map_type(pc, &t, &scale)) return false;
    *out_col = BoltColumn::make_flat_alloc(rows_sum, t, arena);
    if (out_col->data == nullptr) return false;
    out_col->decimal_scale = scale;
    ColCtx cx;
    std::memset(&cx, 0, sizeof(cx));
    cx.pc = pc;
    cx.type = t;
    cx.elem = static_cast<uint32_t>(bolt::type_size(t));
    cx.out = static_cast<uint8_t*>(out_col->data);
    if (pc->optional != 0 && ch->null_count != 0) {
        const uint64_t nb = (static_cast<uint64_t>(rows_sum) + 7u) / 8u;
        uint8_t* bm = static_cast<uint8_t*>(arena->allocate(nb, 1));
        if (bm == nullptr) return false;
        std::memset(bm, 0xFF, nb);                      // pages clear null bits
        out_col->validity = bm;
        out_col->stats.all_valid = false;
        cx.validity = bm;
    }
    cx.arena = arena;
    cx.ovf_base = &out_col->str_overflow_base;
    if (t == BoltType::Utf8 && unc_sum > 0) {
        char* ob = static_cast<char*>(arena->allocate(unc_sum, 1));
        if (ob == nullptr) return false;
        cx.overflow = ob;
        cx.overflow_cap = unc_sum;
        out_col->str_overflow_base = ob;
    }
    // ---- Decode the dictionary, once, before anything references it ----
    void* zscratch = nullptr;                 // lazily allocated, chunk-scoped
    if (has_dict) {
        const uint8_t* dd = buf + dict_pay;
        uint64_t dd_len = static_cast<uint64_t>(dict_hdr.cmp);
        if (!decompress_page(ch->codec, dd, dd_len, dict_hdr.unc, arena,
                             &zscratch, &dd, &dd_len)) {
            return false;
        }
        if (!decode_dict_page(&cx, dd, dd_len, dict_hdr.nvals, arena)) {
            return false;
        }
    }

    // ---- Pass 2: decode the same pages (reuses the whole-chunk page path) ----
    p = p_start;
    int64_t rows_done = 0;
    for (uint32_t page = 0; page < kPqMaxPagesPerChunk; ++page) {
        if (rows_done >= rows_sum) break;
        if (p >= end) break;
        TcCursor c{buf + p, buf + end};
        PageHdr h;
        if (!parse_page_header(&c, &h)) return false;
        const uint64_t pay = p + static_cast<uint64_t>(c.p - (buf + p));
        if (static_cast<uint64_t>(h.cmp) > end - pay) return false;
        const uint8_t* pd = buf + pay;
        uint64_t pd_len = static_cast<uint64_t>(h.cmp);
        if (h.type == kPageData) {
            if (!decompress_page(ch->codec, pd, pd_len, h.unc, arena,
                                 &zscratch, &pd, &pd_len)) {
                return false;
            }
            if (!decode_data_page(&cx, pd, pd_len, &h, rows_done,
                                  rows_sum - rows_done, arena)) {
                return false;
            }
            rows_done += h.nvals;
        } else if (h.type != kPageIndex) {
            return false;
        }
        p = pay + static_cast<uint64_t>(h.cmp);
    }
    if (rows_done != rows_sum) return false;
    *out_rows = rows_sum;
    *next_off = reached_end ? 0 : p_after;
    return true;
}

bool parquet_read_list_column(const uint8_t* buf, uint64_t len,
                              const PqMeta* meta, uint32_t row_group,
                              uint16_t col, Arena* arena,
                              BoltColumn* out_col, int64_t* out_rows) noexcept {
    if (buf == nullptr || meta == nullptr || arena == nullptr) return false;
    if (out_col == nullptr || out_rows == nullptr) return false;
    if (row_group >= meta->n_row_groups) return false;
    if (col >= meta->n_columns) return false;
    const PqColumn* pc = &meta->columns[col];
    if (pc->max_rep == 0u) return false;      // not repeated: use the flat path
    if (pc->max_rep != 1u) {
        // A list of lists needs one offset array per level and an assembly
        // that tracks which level each rep value re-opens. Refused rather
        // than guessed at -- a wrong nesting silently reshapes the data.
        if (bolt_pq_diag()) {
            std::fprintf(stderr, "bolt parquet: column '%s' max_rep=%u -- "
                         "only one level of repetition is supported\n",
                         pc->name, static_cast<unsigned>(pc->max_rep));
        }
        return false;
    }
    return build_list_column(buf, len, meta, row_group, row_group + 1u, col,
                             arena, out_col,
                             out_rows);
}

bool parquet_read_file(const uint8_t* buf, uint64_t len, Arena* arena,
                       BoltBatch* out_batch) noexcept {
    assert(arena != nullptr);
    assert(out_batch != nullptr);
    if (buf == nullptr || len == 0) return false;
    // PqMeta is ~110 KB — arena, never the stack.
    PqMeta* meta = static_cast<PqMeta*>(
        arena->allocate(sizeof(PqMeta), alignof(PqMeta)));
    if (meta == nullptr) return false;
    std::memset(meta, 0, sizeof(*meta));
    if (!parquet_read_meta(buf, len, arena, meta)) return false;
    if (meta->num_rows < 0) return false;
    BoltBatch::init_empty(out_batch);
    // G2FEAT-47: right-size columns[2] before dereferencing columns[read_epoch].
    if (!BoltBatch::alloc_columns(out_batch, arena, meta->n_columns)) return false;
    out_batch->num_rows = meta->num_rows;
    BoltColumn* cols = out_batch->columns[out_batch->read_epoch];
    ColCtx* cxs = arena->allocate_array<ColCtx>(meta->n_columns);
    if (cxs == nullptr) return false;
    // A REPEATED leaf is a list (or a map leaf) and cannot be built by the
    // flat path, which assumes one value per row. Build those through Dremel
    // assembly instead of refusing the file: parquet_read_list_column already
    // decodes them correctly, and refusing here meant a file with a single
    // list<int64> column could not be opened by the obvious call at all.
    bool* is_list = arena->allocate_array<bool>(meta->n_columns);
    if (is_list == nullptr) return false;
    for (uint32_t c = 0; c < meta->n_columns; ++c) {    // bounded: <= kPqMaxColumns (128)
        is_list[c] = (meta->columns[c].max_rep != 0u);
        if (is_list[c]) {
            // ONE level of repetition only. build_list_column is reached
            // directly here, so it does NOT get parquet_read_list_column's
            // max_rep guard -- and without this an unsupported list-of-lists
            // would assemble as if it were a flat list, which is a silent
            // reshape of the data rather than a refusal.
            if (meta->columns[c].max_rep != 1u) {
                if (bolt_pq_diag()) {
                    std::fprintf(stderr, "bolt parquet: column '%s' max_rep=%u"
                                 " -- only one level of repetition is "
                                 "supported\n", meta->columns[c].name,
                                 static_cast<unsigned>(meta->columns[c].max_rep));
                }
                return false;
            }
            int64_t lrows = 0;
            if (!build_list_column(buf, len, meta, 0, meta->n_row_groups, c,
                                   arena, &cols[c], &lrows)) {
                return false;
            }
            if (lrows != meta->num_rows) return false;
            (void)out_batch->schema.add_field(meta->columns[c].name,
                                              BoltType::List);
            continue;
        }
        if (!init_col_ctx(meta, c, 0, meta->n_row_groups, meta->num_rows,
                          arena, &cols[c], &cxs[c])) {
            return false;
        }
        (void)out_batch->schema.add_field(meta->columns[c].name, cxs[c].type);
    }
    int64_t row_off = 0;
    for (uint32_t g = 0; g < meta->n_row_groups; ++g) {  // bounded: <= 4096
        const PqRowGroup* rg = &meta->row_groups[g];
        if (rg->num_rows < 0 || rg->num_rows > meta->num_rows - row_off) {
            return false;
        }
        for (uint32_t c = 0; c < meta->n_columns; ++c) {
            if (is_list[c]) continue;    // already assembled across all groups
            const PqChunk* ch = &meta->chunks[rg->chunk_off + c];
            if (ch->num_values != rg->num_rows) return false;
            if (!decode_chunk(buf, len, ch, &cxs[c], row_off, rg->num_rows,
                              arena)) {
                return false;
            }
        }
        row_off += rg->num_rows;
    }
    assert(row_off <= meta->num_rows);
    return row_off == meta->num_rows;
}

}  // namespace parquet
}  // namespace ingest
}  // namespace bolt
