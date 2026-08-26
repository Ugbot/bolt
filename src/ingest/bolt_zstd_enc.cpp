// bolt_zstd_enc.cpp — self-contained zstd compressor. See the header for
// scope and for why there is no smaller correct version.

#include "bolt/ingest/bolt_zstd_enc.h"

#include <cassert>
#include <cstring>

namespace bolt {
namespace ingest {

namespace {

// ---- tables shared with the decoder ---------------------------------------
//
// Transcribed ONCE, here, and cross-checked against the decoder's copies by
// test_bolt_zstd_enc. The predefined distributions are the one place a silent
// two-entry error already cost this tree a debugging session, so the test
// asserts they agree rather than trusting that they do.
const std::int16_t kLLDefault[36] = {
     4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1,
     2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1,
    -1,-1,-1,-1 };
const std::int16_t kMLDefault[53] = {
     1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,
    -1,-1,-1,-1,-1 };
const std::int16_t kOFDefault[29] = {
     1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1 };

const std::uint32_t kLLBase[36] = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,22,24,28,32,40,48,64,
    128,256,512,1024,2048,4096,8192,16384,32768,65536 };
const std::uint8_t kLLBits[36] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,2,2,3,3,4,6,7,8,9,10,11,12,
    13,14,15,16 };
const std::uint32_t kMLBase[53] = {
    3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,
    29,30,31,32,33,34,35,37,39,41,43,47,51,59,67,83,99,131,259,515,1027,2051,
    4099,8195,16387,32771,65539 };
const std::uint8_t kMLBits[53] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,2,2,3,3,4,4,5,7,8,9,10,11,12,13,14,15,16 };

constexpr std::uint32_t kLLMaxSym = 35;
constexpr std::uint32_t kMLMaxSym = 52;
constexpr std::uint32_t kOFMaxSym = 28;
constexpr std::uint32_t kLLLog = 6;
constexpr std::uint32_t kMLLog = 6;
constexpr std::uint32_t kOFLog = 5;

constexpr std::uint32_t kBlockMax  = 128u * 1024u;
constexpr std::uint32_t kMinMatch  = 3u;
constexpr std::uint32_t kHashBits  = 17u;
constexpr std::uint32_t kHashSize  = 1u << kHashBits;
constexpr std::uint32_t kMaxSeqPerBlock = kBlockMax / 3u + 8u;
constexpr std::uint32_t kNoPos = 0xFFFFFFFFu;

inline std::uint32_t highbit32(std::uint32_t v) noexcept {
    assert(v != 0u);
    std::uint32_t r = 0;
    while (v >>= 1) ++r;
    return r;
}

// ---- FSE encoding table ---------------------------------------------------
//
// Mirrors FSE_buildCTable. `next_state` is the state transition table and
// each symbol carries the two deltas the encode step needs.
struct FseCTable {
    std::uint16_t next_state[1u << 6];
    std::int32_t  delta_nb_bits[64];
    std::int32_t  delta_find_state[64];
    std::uint32_t table_log;
};

bool fse_build_ctable(FseCTable* t, const std::int16_t* norm,
                      std::uint32_t max_sym, std::uint32_t table_log) noexcept {
    assert(t != nullptr && norm != nullptr);
    assert(table_log <= 6u && max_sym < 64u);
    const std::uint32_t size = 1u << table_log;
    const std::uint32_t mask = size - 1u;
    t->table_log = table_log;

    std::uint32_t cumul[66];
    std::uint8_t  sym_of_slot[1u << 6];
    // Low-probability symbols (count -1) are placed from the END of the
    // table downwards; everything else is spread by the step below.
    std::uint32_t high = size - 1u;
    cumul[0] = 0;
    for (std::uint32_t s = 0; s <= max_sym; ++s) {
        if (norm[s] == -1) {
            sym_of_slot[high--] = static_cast<std::uint8_t>(s);
            cumul[s + 1u] = cumul[s] + 1u;
        } else {
            cumul[s + 1u] = cumul[s] + static_cast<std::uint32_t>(norm[s]);
        }
    }
    if (cumul[max_sym + 1u] != size) return false;   // distribution must fill

    const std::uint32_t step = (size >> 1) + (size >> 3) + 3u;
    std::uint32_t pos = 0;
    for (std::uint32_t s = 0; s <= max_sym; ++s) {
        for (std::int32_t i = 0; i < norm[s]; ++i) {
            sym_of_slot[pos] = static_cast<std::uint8_t>(s);
            pos = (pos + step) & mask;
            while (pos > high) pos = (pos + step) & mask;   // skip low-prob tail
        }
    }
    if (pos != 0u) return false;                     // the spread must close

    std::uint32_t c2[66];
    std::memcpy(c2, cumul, sizeof(c2));
    for (std::uint32_t u = 0; u < size; ++u) {
        const std::uint32_t s = sym_of_slot[u];
        t->next_state[c2[s]++] = static_cast<std::uint16_t>(size + u);
    }

    std::uint32_t total = 0;
    for (std::uint32_t s = 0; s <= max_sym; ++s) {
        const std::int16_t n = norm[s];
        if (n == 0) {
            t->delta_nb_bits[s] =
                static_cast<std::int32_t>(((table_log + 1u) << 16) - (1u << table_log));
            t->delta_find_state[s] = 0;
        } else if (n == -1 || n == 1) {
            t->delta_nb_bits[s] =
                static_cast<std::int32_t>((table_log << 16) - (1u << table_log));
            t->delta_find_state[s] = static_cast<std::int32_t>(total) - 1;
            total += 1u;
        } else {
            const std::uint32_t max_bits =
                table_log - highbit32(static_cast<std::uint32_t>(n) - 1u);
            const std::uint32_t min_state_plus =
                static_cast<std::uint32_t>(n) << max_bits;
            t->delta_nb_bits[s] =
                static_cast<std::int32_t>((max_bits << 16) - min_state_plus);
            t->delta_find_state[s] =
                static_cast<std::int32_t>(total) - static_cast<std::int32_t>(n);
            total += static_cast<std::uint32_t>(n);
        }
    }
    return true;
}

// ---- forward bit writer ---------------------------------------------------
//
// zstd's sequence bitstream is written FORWARD but read BACKWARD: the decoder
// starts at the last byte and walks down, which is why the stream ends with a
// set marker bit locating the final partial byte.
struct BitC {
    std::uint8_t* dst;
    std::uint64_t cap;
    std::uint64_t pos;
    std::uint64_t container;
    std::uint32_t bits;
    bool          ok;
};

inline void bitc_add(BitC* b, std::uint32_t value, std::uint32_t n) noexcept {
    if (n == 0u) return;
    assert(n <= 32u);
    const std::uint64_t m =
        (n >= 32u) ? 0xFFFFFFFFull : ((1ull << n) - 1ull);
    b->container |= (static_cast<std::uint64_t>(value) & m) << b->bits;
    b->bits += n;
}

inline void bitc_flush(BitC* b) noexcept {
    const std::uint32_t nbytes = b->bits >> 3;
    if (b->pos + 8u > b->cap) { b->ok = false; return; }
    std::memcpy(b->dst + b->pos, &b->container, 8);   // little-endian target
    b->pos += nbytes;
    b->bits &= 7u;
    b->container >>= (nbytes * 8u);
}

// Returns the stream length, or 0 on overflow.
inline std::uint64_t bitc_close(BitC* b) noexcept {
    bitc_add(b, 1u, 1u);                              // end marker
    bitc_flush(b);
    if (!b->ok) return 0;
    return b->pos + ((b->bits > 0u) ? 1u : 0u);
}

struct FseCState { std::uint32_t value; const FseCTable* t; };

inline void fse_init_cstate(FseCState* s, const FseCTable* t,
                            std::uint32_t sym) noexcept {
    s->t = t;
    const std::uint32_t nb =
        static_cast<std::uint32_t>((t->delta_nb_bits[sym] + (1 << 15)) >> 16);
    const std::uint32_t v =
        (nb << 16) - static_cast<std::uint32_t>(t->delta_nb_bits[sym]);
    s->value = t->next_state[(v >> nb) +
                             static_cast<std::uint32_t>(t->delta_find_state[sym])];
}

inline void fse_encode(BitC* b, FseCState* s, std::uint32_t sym) noexcept {
    const std::uint32_t nb = static_cast<std::uint32_t>(
        (static_cast<std::int32_t>(s->value) + s->t->delta_nb_bits[sym]) >> 16);
    bitc_add(b, s->value, nb);
    s->value = s->t->next_state[(s->value >> nb) +
        static_cast<std::uint32_t>(s->t->delta_find_state[sym])];
}

inline void fse_flush_cstate(BitC* b, const FseCState* s) noexcept {
    bitc_add(b, s->value, s->t->table_log);
    bitc_flush(b);
}

// ---- sequence code derivation ---------------------------------------------
//
// Derived from the DECODER's base/bits tables so the two cannot disagree.
inline std::uint32_t code_for(std::uint32_t value, const std::uint32_t* base,
                              std::uint32_t n) noexcept {
    std::uint32_t c = n - 1u;
    while (c > 0u && base[c] > value) --c;
    return c;
}

struct Seq {
    std::uint32_t lit_len;
    std::uint32_t match_len;   // real match length (>= kMinMatch)
    std::uint32_t off_base;    // offset + 3
};

}  // namespace

struct ZstdEncState {
    std::uint32_t table[kHashSize];
    Seq           seqs[kMaxSeqPerBlock];
    std::uint8_t  lits[kBlockMax + 32u];
    std::uint8_t  seqbuf[kBlockMax + 1024u];
};

std::uint64_t zstd_enc_state_size() noexcept { return sizeof(ZstdEncState); }

ZstdEncState* zstd_enc_state_init(void* mem, std::uint64_t len) noexcept {
    if (mem == nullptr || len < sizeof(ZstdEncState)) return nullptr;
    return static_cast<ZstdEncState*>(mem);
}

namespace {

inline std::uint32_t load32(const std::uint8_t* p) noexcept {
    std::uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

inline std::uint32_t hash4(std::uint32_t v) noexcept {
    return (v * 2654435761u) >> (32u - kHashBits);
}

// Compress one block's worth of input into `out`. Returns the number of bytes
// written, or 0 to mean "not worth compressing; the caller should store raw".
std::uint64_t compress_block(const std::uint8_t* src, std::uint32_t n,
                             std::uint8_t* out, std::uint64_t cap,
                             ZstdEncState* st) noexcept {
    if (n < 16u) return 0;
    std::memset(st->table, 0xFF, sizeof(st->table));

    std::uint32_t nseq = 0;
    std::uint32_t nlit = 0;
    std::uint32_t anchor = 0;
    std::uint32_t ip = 1;                       // never match at 0
    const std::uint32_t limit = n - 12u;
    while (ip < limit && nseq + 1u < kMaxSeqPerBlock) {
        const std::uint32_t h = hash4(load32(src + ip));
        const std::uint32_t cand = st->table[h];
        st->table[h] = ip;
        if (cand == kNoPos || ip - cand > 65535u ||
            load32(src + cand) != load32(src + ip)) {
            ++ip;
            continue;
        }
        std::uint32_t mlen = 4u;
        while (ip + mlen < n && src[cand + mlen] == src[ip + mlen]) ++mlen;
        const std::uint32_t ll = ip - anchor;
        if (nlit + ll > kBlockMax) return 0;
        std::memcpy(st->lits + nlit, src + anchor, ll);
        nlit += ll;
        st->seqs[nseq].lit_len = ll;
        st->seqs[nseq].match_len = mlen;
        st->seqs[nseq].off_base = (ip - cand) + 3u;
        ++nseq;
        ip += mlen;
        anchor = ip;
    }
    if (nseq == 0u) return 0;                   // nothing found: store raw
    const std::uint32_t tail = n - anchor;
    if (nlit + tail > kBlockMax) return 0;
    std::memcpy(st->lits + nlit, src + anchor, tail);
    nlit += tail;

    // ---- literals section: Raw_Literals_Block ----
    std::uint64_t op = 0;
    if (nlit < 32u) {                           // size_format 00: 1 byte
        if (op + 1u > cap) return 0;
        out[op++] = static_cast<std::uint8_t>(nlit << 3);
    } else if (nlit < 4096u) {                  // size_format 01: 2 bytes
        if (op + 2u > cap) return 0;
        out[op++] = static_cast<std::uint8_t>(0x4u | ((nlit & 0xFu) << 4));
        out[op++] = static_cast<std::uint8_t>(nlit >> 4);
    } else {                                    // size_format 11: 3 bytes
        // Regenerated_Size is 20 bits spanning bits 4..23 -- the low nibble
        // shares byte 0 with the type and size format. An earlier version
        // split it 2/8/8 instead of 4/8/8 and produced frames libzstd
        // rejected as "Data corruption detected". It survived the reference
        // check because this path needs a block that BOTH compresses and has
        // a literal run of 4096+, and the synthetic corpus had neither
        // together: random blocks store raw (so no literals header at all)
        // and the repetitive ones have tiny literal runs.
        if (op + 3u > cap) return 0;
        out[op++] = static_cast<std::uint8_t>(0xCu | ((nlit & 0xFu) << 4));
        out[op++] = static_cast<std::uint8_t>((nlit >> 4) & 0xFFu);
        out[op++] = static_cast<std::uint8_t>((nlit >> 12) & 0xFFu);
    }
    if (op + nlit > cap) return 0;
    std::memcpy(out + op, st->lits, nlit);
    op += nlit;

    // ---- sequences section header ----
    if (nseq < 128u) {
        if (op + 1u > cap) return 0;
        out[op++] = static_cast<std::uint8_t>(nseq);
    } else if (nseq < 0x7F00u) {
        if (op + 2u > cap) return 0;
        out[op++] = static_cast<std::uint8_t>((nseq >> 8) + 0x80u);
        out[op++] = static_cast<std::uint8_t>(nseq & 0xFFu);
    } else {
        if (op + 3u > cap) return 0;
        out[op++] = 0xFFu;
        const std::uint32_t v = nseq - 0x7F00u;
        out[op++] = static_cast<std::uint8_t>(v & 0xFFu);
        out[op++] = static_cast<std::uint8_t>(v >> 8);
    }
    // Symbol_Compression_Modes: predefined for all three (bits 6-7 LL,
    // 4-5 OF, 2-3 ML; 0 == Predefined_Mode), so no table is transmitted.
    if (op + 1u > cap) return 0;
    out[op++] = 0u;

    FseCTable ll_t, ml_t, of_t;
    if (!fse_build_ctable(&ll_t, kLLDefault, kLLMaxSym, kLLLog)) return 0;
    if (!fse_build_ctable(&ml_t, kMLDefault, kMLMaxSym, kMLLog)) return 0;
    if (!fse_build_ctable(&of_t, kOFDefault, kOFMaxSym, kOFLog)) return 0;

    // ---- the bitstream ----
    //
    // Written forward, read backward, so everything goes in REVERSE sequence
    // order and the final flush order is the reverse of the decoder's init
    // order (it reads LL, OF, ML; the last thing written must therefore be
    // LL). Within a sequence the decoder reads offset, match, literal -- so
    // they are written literal, match, offset.
    BitC b{st->seqbuf, sizeof(st->seqbuf), 0, 0, 0, true};
    const std::uint32_t last = nseq - 1u;
    const std::uint32_t ll_c0 = code_for(st->seqs[last].lit_len, kLLBase, 36u);
    const std::uint32_t ml_c0 = code_for(st->seqs[last].match_len, kMLBase, 53u);
    const std::uint32_t of_c0 = highbit32(st->seqs[last].off_base);
    if (of_c0 > kOFMaxSym) return 0;

    FseCState s_ml, s_of, s_ll;
    fse_init_cstate(&s_ml, &ml_t, ml_c0);
    fse_init_cstate(&s_of, &of_t, of_c0);
    fse_init_cstate(&s_ll, &ll_t, ll_c0);
    bitc_add(&b, st->seqs[last].lit_len - kLLBase[ll_c0], kLLBits[ll_c0]);
    bitc_flush(&b);
    bitc_add(&b, st->seqs[last].match_len - kMLBase[ml_c0], kMLBits[ml_c0]);
    bitc_flush(&b);
    bitc_add(&b, st->seqs[last].off_base - (1u << of_c0), of_c0);
    bitc_flush(&b);

    for (std::int64_t i = static_cast<std::int64_t>(last) - 1; i >= 0; --i) {
        const Seq& s = st->seqs[i];
        const std::uint32_t llc = code_for(s.lit_len, kLLBase, 36u);
        const std::uint32_t mlc = code_for(s.match_len, kMLBase, 53u);
        const std::uint32_t ofc = highbit32(s.off_base);
        if (ofc > kOFMaxSym) return 0;
        fse_encode(&b, &s_of, ofc);
        fse_encode(&b, &s_ml, mlc);
        fse_encode(&b, &s_ll, llc);
        bitc_flush(&b);
        bitc_add(&b, s.lit_len - kLLBase[llc], kLLBits[llc]);
        bitc_add(&b, s.match_len - kMLBase[mlc], kMLBits[mlc]);
        bitc_flush(&b);
        bitc_add(&b, s.off_base - (1u << ofc), ofc);
        bitc_flush(&b);
    }
    fse_flush_cstate(&b, &s_ml);
    fse_flush_cstate(&b, &s_of);
    fse_flush_cstate(&b, &s_ll);
    const std::uint64_t blen = bitc_close(&b);
    if (blen == 0u || !b.ok) return 0;
    if (op + blen > cap) return 0;
    std::memcpy(out + op, st->seqbuf, blen);
    op += blen;
    // Only worth it if it actually shrank the block.
    return (op < n) ? op : 0;
}

}  // namespace

bool zstd_compress_self(const std::uint8_t* src, std::uint64_t src_len,
                        std::uint8_t* dst, std::uint64_t dst_cap,
                        std::uint64_t* out_len, ZstdEncState* st) noexcept {
    assert(out_len != nullptr);
    assert(src != nullptr || src_len == 0);
    if (dst == nullptr || st == nullptr) return false;

    std::uint64_t op = 0;
    // Frame header: magic, then a descriptor with Single_Segment set (so the
    // window is the content size and no Window_Descriptor follows) and an
    // FCS field sized to the content.
    if (op + 4u > dst_cap) return false;
    dst[op++] = 0x28; dst[op++] = 0xB5; dst[op++] = 0x2F; dst[op++] = 0xFD;
    std::uint32_t fcs_flag;
    std::uint32_t fcs_bytes;
    if (src_len < 256u)              { fcs_flag = 0u; fcs_bytes = 1u; }
    else if (src_len < 65536u + 256u){ fcs_flag = 1u; fcs_bytes = 2u; }
    else if (src_len <= 0xFFFFFFFFu) { fcs_flag = 2u; fcs_bytes = 4u; }
    else                             { fcs_flag = 3u; fcs_bytes = 8u; }
    if (op + 1u + fcs_bytes > dst_cap) return false;
    dst[op++] = static_cast<std::uint8_t>((fcs_flag << 6) | (1u << 5));
    if (fcs_bytes == 2u) {
        const std::uint16_t v = static_cast<std::uint16_t>(src_len - 256u);
        dst[op++] = static_cast<std::uint8_t>(v & 0xFFu);
        dst[op++] = static_cast<std::uint8_t>(v >> 8);
    } else {
        for (std::uint32_t i = 0; i < fcs_bytes; ++i) {
            dst[op++] = static_cast<std::uint8_t>((src_len >> (8u * i)) & 0xFFu);
        }
    }

    std::uint64_t off = 0;
    do {
        const std::uint32_t n = (src_len - off > kBlockMax)
            ? kBlockMax : static_cast<std::uint32_t>(src_len - off);
        const bool last = (off + n >= src_len);
        // Try compressed; fall back to a Raw block, which is always legal and
        // is what incompressible input must take (a compressed block that is
        // bigger than its input is a loss, not a bug).
        std::uint64_t clen = 0;
        if (op + 3u <= dst_cap) {
            clen = compress_block(src + off, n, dst + op + 3u,
                                  dst_cap - op - 3u, st);
        }
        const bool use_comp = (clen > 0u);
        const std::uint32_t bsize = use_comp ? static_cast<std::uint32_t>(clen) : n;
        const std::uint32_t btype = use_comp ? 2u : 0u;
        if (op + 3u + bsize > dst_cap) return false;
        const std::uint32_t hdr = (last ? 1u : 0u) | (btype << 1) | (bsize << 3);
        dst[op++] = static_cast<std::uint8_t>(hdr & 0xFFu);
        dst[op++] = static_cast<std::uint8_t>((hdr >> 8) & 0xFFu);
        dst[op++] = static_cast<std::uint8_t>((hdr >> 16) & 0xFFu);
        if (!use_comp) std::memcpy(dst + op, src + off, n);
        op += bsize;
        off += n;
    } while (off < src_len);

    // No special case for an empty frame: the do-while above already runs
    // once with n == 0 and emits a last Raw block of size 0, which is the
    // correct encoding. Emitting another one here put a second block AFTER
    // the one flagged last, and libzstd rejected the frame as "Src size is
    // incorrect" -- the only 5 of 40 reference frames that failed.
    *out_len = op;
    return true;
}

}  // namespace ingest
}  // namespace bolt
