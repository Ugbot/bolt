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
#include <cstring>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/ingest/bolt_snappy.h"

namespace bolt {
namespace ingest {
namespace parquet {

namespace {

constexpr uint32_t kPqMaxPagesPerChunk = 1u << 16;
constexpr uint32_t kPqMaxDictEntries   = 1u << 24;
constexpr int64_t  kPqMaxPageBytes     = int64_t{1} << 30;
constexpr uint32_t kPqMetaChunksFirst  = 4096;

// parquet::PageType / Encoding / ConvertedType values we consume.
constexpr int32_t kPageData = 0, kPageIndex = 1, kPageDict = 2;
constexpr int32_t kEncPlain = 0, kEncPlainDict = 2, kEncRle = 3,
                  kEncRleDict = 8;
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
    for (uint32_t i = 0; i < n; ++i) {
        const uint64_t bit  = static_cast<uint64_t>(i) * bw;
        const uint64_t byte = bit >> 3;
        const uint32_t sh   = static_cast<uint32_t>(bit & 7u);
        uint64_t w = 0;
        if (byte + 8 <= in_len) {
            std::memcpy(&w, in + byte, 8);
        } else {
            uint8_t tmp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            assert(byte <= in_len);   // caller pre-checked total payload
            std::memcpy(tmp, in + byte, in_len - byte);
            std::memcpy(&w, tmp, 8);
        }
        out[i] = static_cast<uint32_t>((w >> sh) & mask);
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
//             7 dictionary_page_header} (4 crc / 8 v2 header skipped).
bool parse_page_header(TcCursor* c, PageHdr* h) noexcept {
    assert(c != nullptr && h != nullptr);
    h->type = -1; h->unc = -1; h->cmp = -1;
    h->nvals = -1; h->enc = -1; h->def_enc = -1;
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
    uint32_t elem;              // output stride: 4 / 8 / 16
    uint8_t* out;               // column data base
    uint8_t* validity;          // nullptr => no bitmap (all-valid column)
    char*    overflow;          // Utf8 spill buffer (column-wide)
    uint64_t overflow_cap;
    uint64_t overflow_cursor;
    uint8_t* dict;              // decoded dict entries, elem stride
    uint32_t dict_n;
};

// Build a StringView; >12-byte payloads spill into the column overflow.
bool sv_from_bytes(ColCtx* cx, const uint8_t* p, uint32_t len,
                   StringView* out) noexcept {
    assert(cx != nullptr && out != nullptr);
    assert(p != nullptr || len == 0);
    std::memset(out, 0, sizeof(*out));
    out->length = len;
    if (len == 0) return true;
    if (len <= 12u) {
        const uint32_t pfx = (len < 4u) ? len : 4u;
        std::memcpy(out->prefix, p, pfx);
        if (len > 4u) std::memcpy(out->inline_data, p + 4, len - 4u);
        return true;
    }
    if (cx->overflow == nullptr) return false;
    const uint64_t cur = cx->overflow_cursor;
    if (cur + len > cx->overflow_cap) return false;
    if (cur > 0xFFFFFFFFull) return false;       // StringView::ref.offset u32
    std::memcpy(out->prefix, p, 4);
    std::memcpy(cx->overflow + cur, p, len);
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
        if (def[i] == 0u) { bit_clear(cx->validity, row0 + i); continue; }
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
        if (def != nullptr && def[i] == 0u) {
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
        if (def != nullptr && def[i] == 0u) {
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
        if (def != nullptr && def[i] == 0u) {
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
        if (def != nullptr && def[i] == 0u) {
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
        if (def != nullptr && def[i] == 0u) {
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
bool dict_gather(ColCtx* cx, const uint32_t* idx, const uint32_t* def,
                 int64_t row0, uint32_t nrows, uint32_t nvalid) noexcept {
    assert(cx != nullptr && cx->out != nullptr);
    if (cx->dict == nullptr) return false;
    const uint32_t w = cx->elem;
    uint32_t k = 0;
    for (uint32_t i = 0; i < nrows; ++i) {
        if (def != nullptr && def[i] == 0u) {
            bit_clear(cx->validity, row0 + i);
            continue;
        }
        if (k >= nvalid) return false;
        const uint32_t id = idx[k++];
        if (id >= cx->dict_n) return false;
        std::memcpy(cx->out + (static_cast<uint64_t>(row0) + i) * w,
                    cx->dict + static_cast<uint64_t>(id) * w, w);
    }
    return k == nvalid;
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
        case PqType::Double:
            ok = plain_fixed(&tmp, v, vlen, nullptr, 0, n, n);
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

// DATA_PAGE (v1): [def levels if OPTIONAL: u32 LE byte-len + RLE hybrid]
// then PLAIN values or [bit-width byte + RLE hybrid dictionary indices].
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
        if (vlen < 4) return false;
        uint32_t dl = 0;
        std::memcpy(&dl, v, 4);
        if (4ull + dl > vlen) return false;
        def = static_cast<uint32_t*>(
            arena->allocate(uint64_t{nvals} * 4u, 4));
        if (def == nullptr) return false;
        if (!rle_hybrid_decode(v + 4, dl, 1, nvals, def)) return false;
        v += 4ull + dl;
        vlen -= 4ull + dl;
        nvalid = 0;
        for (uint32_t i = 0; i < nvals; ++i) nvalid += def[i];
        if (nvalid == nvals) def = nullptr;         // all-valid: dense path
        if (def != nullptr && cx->validity == nullptr) {
            return false;                           // chunk stats said no nulls
        }
    }
    if (h->enc == kEncPlain) {
        switch (cx->pc->physical) {
            case PqType::Int32:
            case PqType::Int64:
            case PqType::Double:
                return plain_fixed(cx, v, vlen, def, row0, nvals, nvalid);
            case PqType::Float:
                return plain_f32_widen(cx, v, vlen, def, row0, nvals,
                                       nvalid);
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
    if (h->enc == kEncRle && cx->pc->physical == PqType::Boolean) {
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
        if (ch->codec == PqCodec::Snappy) {
            uint8_t* dst = static_cast<uint8_t*>(
                arena->allocate(static_cast<uint64_t>(h.unc) + 1u, 8));
            if (dst == nullptr) return false;
            if (!snappy_decompress(pd, pd_len, dst,
                                   static_cast<uint64_t>(h.unc))) {
                return false;
            }
            pd = dst;
            pd_len = static_cast<uint64_t>(h.unc);
        } else if (ch->codec == PqCodec::Uncompressed) {
            if (h.cmp != h.unc) return false;
        } else {
            return false;                     // codec outside v1
        }
        if (h.type == kPageDict) {
            if (!decode_dict_page(cx, pd, pd_len, h.nvals, arena)) {
                return false;
            }
        } else if (h.type == kPageData) {
            if (!decode_data_page(cx, pd, pd_len, &h, row0 + rows_done,
                                  rows - rows_done, arena)) {
                return false;
            }
            rows_done += h.nvals;
        } else if (h.type != kPageIndex) {
            return false;                     // DATA_PAGE_V2 etc.: rejected
        }
        p = pay + static_cast<uint64_t>(h.cmp);
    }
    return rows_done == rows;
}

// Allocate one output column + its decode context for chunk range
// [g0, g1) of column c. `rows` = total rows the column will hold.
bool init_col_ctx(const PqMeta* m, uint32_t c, uint32_t g0, uint32_t g1,
                  int64_t rows, Arena* arena, BoltColumn* col,
                  ColCtx* cx) noexcept {
    assert(m != nullptr && cx != nullptr);
    assert(c < m->n_columns && col != nullptr);
    const PqColumn* pc = &m->columns[c];
    BoltType t;
    uint8_t scale = 0;
    if (!parquet_map_type(pc, &t, &scale)) return false;
    *col = BoltColumn::make_flat_alloc(rows, t, arena);
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
    cx->elem = static_cast<uint32_t>(bolt::type_size(t));
    cx->out = static_cast<uint8_t*>(col->data);
    cx->validity = col->validity;
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

// ---- public API ---------------------------------------------------------------

bool parquet_map_type(const PqColumn* col, BoltType* out_type,
                      uint8_t* out_scale) noexcept {
    assert(out_type != nullptr);
    assert(out_scale != nullptr);
    if (col == nullptr) return false;
    *out_scale = 0;
    switch (col->physical) {
        case PqType::Int64:
            if (col->converted == kConvDecimal) {       // int64 mantissa as-is
                if (col->scale < 0 || col->scale > 18) return false;
                *out_type = BoltType::Decimal64;
                *out_scale = static_cast<uint8_t>(col->scale);
                return true;
            }
            *out_type = BoltType::Int64;
            return true;
        case PqType::Int32:
            if (col->converted == kConvDecimal) return false;   // v1: reject
            *out_type = (col->converted == kConvDate) ? BoltType::Date32
                                                      : BoltType::Int32;
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
            if (col->converted != kConvDecimal) return false;
            if (col->type_length <= 0 || col->type_length > 16) return false;
            if (col->scale < 0 || col->scale > 38) return false;
            if (col->precision <= 18) {                 // W-DEC representation
                *out_type = BoltType::Decimal64;
            } else {
                *out_type = BoltType::Decimal128;
            }
            *out_scale = static_cast<uint8_t>(col->scale);
            return true;
        default:                       // Int96: outside v1
            return false;
    }
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
    for (uint32_t c = 0; c < meta->n_columns; ++c) {    // bounded: <= 64
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
    if (ch->dictionary_page_offset > 0) return false;   // v1: PLAIN only
    if (ch->data_page_offset <= 0 || ch->total_compressed_size <= 0) return false;
    const uint64_t region_start = static_cast<uint64_t>(ch->data_page_offset);
    const uint64_t end =
        region_start + static_cast<uint64_t>(ch->total_compressed_size);
    if (end > len) return false;
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
    if (t == BoltType::Utf8 && unc_sum > 0) {
        char* ob = static_cast<char*>(arena->allocate(unc_sum, 1));
        if (ob == nullptr) return false;
        cx.overflow = ob;
        cx.overflow_cap = unc_sum;
        out_col->str_overflow_base = ob;
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
            if (ch->codec == PqCodec::Snappy) {
                uint8_t* dst = static_cast<uint8_t*>(
                    arena->allocate(static_cast<uint64_t>(h.unc) + 1u, 8));
                if (dst == nullptr) return false;
                if (!snappy_decompress(pd, pd_len, dst,
                                       static_cast<uint64_t>(h.unc))) {
                    return false;
                }
                pd = dst;
                pd_len = static_cast<uint64_t>(h.unc);
            } else if (ch->codec == PqCodec::Uncompressed) {
                if (h.cmp != h.unc) return false;
            } else {
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
    for (uint32_t c = 0; c < meta->n_columns; ++c) {    // bounded: <= 64
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
