// bolt/ingest/bolt_zstd_dec.cpp — self-contained Zstandard decoder (RFC 8878).
//
// Structure, bottom-up:
//   1. bit readers      — forward (FSE header) and backward (entropy streams)
//   2. FSE              — normalized-count reader, decode-table builder
//   3. Huffman          — weight reader (direct + FSE-coded), table builder,
//                         1-stream and 4-stream literal decode
//   4. literals section — Raw / RLE / Compressed / Treeless
//   5. sequences        — 3 interleaved FSE states, repeat offsets
//   6. sequence exec    — literal copy + (overlapping) match copy
//   7. block / frame    — headers, xxhash64 content checksum
//
// SAFETY MODEL (this parses untrusted bytes):
//   * Every read from `src` is bounds-checked against src_len before the read.
//   * Every write to `dst` is bounds-checked against dst_cap before the write.
//   * Over-PEEKING a backward bitstream past its start is legal (zstd's final
//     symbols do it by construction) and shifts in zeros. Over-CONSUMING is
//     reported by zb_overrun(); it is corruption everywhere the symbol count is
//     known in advance, and the normal end-of-stream signal in the Huffman
//     weight decoder. See zb_overrun's comment.
//   * asserts state programmer invariants only — never input validity.

#include "bolt/ingest/bolt_zstd_dec.h"

#include <cassert>
#include <cstring>

namespace bolt {
namespace ingest {
namespace {

// ---------------------------------------------------------------------------
// Bounded caps (Tiger Style: every buffer has a constexpr ceiling)
// ---------------------------------------------------------------------------
constexpr uint32_t kBlockMax      = 1u << 17;  // 128 KiB, zstd's block ceiling
constexpr uint32_t kHufMaxBits    = 11;        // spec: Huffman codes <= 11 bits
constexpr uint32_t kHufTableSize  = 1u << kHufMaxBits;
constexpr uint32_t kMaxSymbols    = 256;
constexpr uint32_t kLLMaxLog      = 9;
constexpr uint32_t kMLMaxLog      = 9;
constexpr uint32_t kOFMaxLog      = 8;
constexpr uint32_t kFseMaxLog     = 9;
constexpr uint32_t kFseTableSize  = 1u << kFseMaxLog;
constexpr uint32_t kWeightLog     = 6;         // Huffman-weight FSE accuracy
constexpr uint32_t kWeightTblSize = 1u << kWeightLog;
constexpr uint32_t kLLSymbolMax   = 35;
constexpr uint32_t kMLSymbolMax   = 52;
constexpr uint32_t kOFSymbolMax   = 31;
// A frame is a bounded number of blocks; guards against a crafted infinite
// stream of zero-length blocks.
constexpr uint32_t kMaxBlocksPerFrame = 1u << 20;
constexpr uint32_t kMaxFrames         = 1u << 16;

inline uint32_t highbit32(uint32_t v) noexcept {
    assert(v != 0);
    uint32_t r = 0;
    while (v >>= 1) ++r;
    return r;
}

// ---------------------------------------------------------------------------
// 1a. Forward bit reader — LSB-first. Used only by the FSE normalized-count
//     header, which is a forward stream.
// ---------------------------------------------------------------------------
struct FBits {
    const uint8_t* p;
    uint64_t       len;      // bytes
    uint64_t       bitpos;   // next bit index
    bool           bad;
};

void fb_init(FBits* b, const uint8_t* p, uint64_t len) noexcept {
    assert(b != nullptr);
    b->p = p; b->len = len; b->bitpos = 0; b->bad = (len == 0);
}

// Peek n (<= 24) bits without consuming. Past-the-end reads as zeros; the
// caller detects real overrun via fb_skip's bounds check.
uint32_t fb_peek(FBits* b, uint32_t n) noexcept {
    assert(n <= 24);
    if (n == 0) return 0;
    const uint64_t byte = b->bitpos >> 3;
    const uint32_t sh   = static_cast<uint32_t>(b->bitpos & 7u);
    uint64_t acc = 0;
    for (uint32_t i = 0; i < 5; ++i) {
        const uint64_t idx = byte + i;
        const uint64_t v = (idx < b->len) ? b->p[idx] : 0u;
        acc |= v << (8u * i);
    }
    return static_cast<uint32_t>((acc >> sh) & ((1u << n) - 1u));
}

void fb_skip(FBits* b, uint32_t n) noexcept {
    b->bitpos += n;
    if (b->bitpos > b->len * 8u) b->bad = true;
}

uint32_t fb_read(FBits* b, uint32_t n) noexcept {
    const uint32_t v = fb_peek(b, n);
    fb_skip(b, n);
    return v;
}

// ---------------------------------------------------------------------------
// 1b. Backward bit reader — MSB-first from the end of the buffer. This is how
//     zstd stores every entropy stream (Huffman literals, FSE sequences).
//
//     The last byte carries a "1" marker bit above the payload; the bits above
//     it are padding and are skipped at init.
// ---------------------------------------------------------------------------
struct ZBits {
    const uint8_t* p;
    int64_t        next;      // index of next byte to pull (descending)
    uint64_t       acc;       // low `bits_in` bits are valid, MSB-first order
    int32_t        bits_in;
    int64_t        avail;     // usable payload bits not yet consumed
    bool           bad;
};

bool zb_init(ZBits* b, const uint8_t* p, uint64_t len) noexcept {
    assert(b != nullptr);
    b->p = p; b->acc = 0; b->bits_in = 0; b->avail = 0;
    b->next = -1; b->bad = true;
    if (len == 0) return false;
    const uint8_t last = p[len - 1];
    if (last == 0) return false;             // missing marker bit => corrupt
    uint32_t lz = 0;
    while (((last << lz) & 0x80u) == 0) ++lz;  // terminates: last != 0
    const int32_t usable = static_cast<int32_t>(7u - lz);
    b->acc     = static_cast<uint64_t>(last) & ((1u << usable) - 1u);
    b->bits_in = usable;
    b->avail   = static_cast<int64_t>(len) * 8 -
                 static_cast<int64_t>(lz) - 1;
    b->next    = static_cast<int64_t>(len) - 2;
    b->bad     = false;
    return true;
}

inline void zb_fill(ZBits* b, uint32_t n) noexcept {
    // Over-peeking past the stream start is legal and reads as zeros.
    while (b->bits_in < static_cast<int32_t>(n)) {
        const uint64_t byte = (b->next >= 0)
                            ? static_cast<uint64_t>(b->p[b->next--]) : 0u;
        b->acc = (b->acc << 8) | byte;
        b->bits_in += 8;
    }
}

inline uint32_t zb_peek(ZBits* b, uint32_t n) noexcept {
    if (n == 0) return 0;
    assert(n <= 32);
    zb_fill(b, n);
    const uint64_t mask = (n >= 32) ? 0xFFFFFFFFull : ((1ull << n) - 1ull);
    return static_cast<uint32_t>((b->acc >> (b->bits_in - static_cast<int32_t>(n))) & mask);
}

inline void zb_skip(ZBits* b, uint32_t n) noexcept {
    b->bits_in -= static_cast<int32_t>(n);
    b->avail   -= static_cast<int64_t>(n);
}

// True once more bits have been consumed than the payload holds.
//
// This is NOT always corruption. Where the symbol count is known up front
// (Huffman literal streams, sequences) an overrun means a malformed stream and
// callers reject it. But the FSE-coded Huffman WEIGHT stream is decoded until
// exhaustion, and the spec says the final state update may need more bits than
// remain, in which case "it is assumed the extra bits are 0" -- there, overrun
// is the normal termination signal. So the flag is reported, not enforced here.
inline bool zb_overrun(const ZBits* b) noexcept {
    return b->bad || b->avail < 0;
}

inline uint32_t zb_read(ZBits* b, uint32_t n) noexcept {
    const uint32_t v = zb_peek(b, n);
    zb_skip(b, n);
    return v;
}

// ---------------------------------------------------------------------------
// 2. FSE
// ---------------------------------------------------------------------------
struct FseTable {
    uint8_t  symbol[kFseTableSize];
    uint8_t  nbits[kFseTableSize];
    uint16_t newstate[kFseTableSize];
    uint32_t log;
};

// Read a normalized-count table (RFC 8878 4.1.1). Returns bytes consumed, or
// 0 on corruption.
uint32_t fse_read_ncount(const uint8_t* src, uint64_t src_len,
                         int16_t* norm, uint32_t max_sv,
                         uint32_t* out_log, uint32_t max_log) noexcept {
    assert(norm != nullptr);
    assert(out_log != nullptr);
    if (src_len < 1) return 0;
    FBits fb; fb_init(&fb, src, src_len);
    const uint32_t log = fb_read(&fb, 4) + 5u;
    if (log > max_log || fb.bad) return 0;
    *out_log = log;

    int32_t  remaining = static_cast<int32_t>(1u << log) + 1;
    int32_t  threshold = static_cast<int32_t>(1u << log);
    uint32_t nbits     = log + 1u;
    uint32_t sym       = 0;
    bool     prev0     = false;

    for (uint32_t i = 0; i <= max_sv; ++i) norm[i] = 0;

    while (remaining > 1 && sym <= max_sv) {
        if (prev0) {
            uint32_t zeros = 0;
            for (uint32_t g = 0; g < 64; ++g) {      // bounded
                const uint32_t r = fb_read(&fb, 2);
                zeros += r;
                if (r != 3) break;
            }
            if (fb.bad) return 0;
            if (zeros > max_sv || sym + zeros > max_sv + 1u) return 0;
            for (uint32_t z = 0; z < zeros; ++z) norm[sym++] = 0;
            prev0 = false;
            if (sym > max_sv) break;
        }
        const int32_t  max = (2 * threshold - 1) - remaining;
        const uint32_t low = fb_peek(&fb, nbits - 1u);
        int32_t count;
        if (static_cast<int32_t>(low) < max) {
            count = static_cast<int32_t>(low);
            fb_skip(&fb, nbits - 1u);
        } else {
            uint32_t v = fb_peek(&fb, nbits);
            fb_skip(&fb, nbits);
            if (static_cast<int32_t>(v) >= threshold) {
                count = static_cast<int32_t>(v) - max;
            } else {
                count = static_cast<int32_t>(v);
            }
        }
        if (fb.bad) return 0;
        count -= 1;                        // -1 encodes "low probability"
        const int32_t weight = (count < 0) ? -count : count;
        remaining -= weight;
        if (remaining < 0) return 0;
        norm[sym++] = static_cast<int16_t>(count);
        prev0 = (count == 0);
        while (remaining < threshold) {
            if (nbits == 0) return 0;
            --nbits;
            threshold >>= 1;
        }
    }
    if (remaining != 1) return 0;
    if (fb.bad) return 0;
    const uint64_t used = (fb.bitpos + 7u) / 8u;
    if (used == 0 || used > src_len) return 0;
    return static_cast<uint32_t>(used);
}

// Build an FSE decode table from normalized counts (RFC 8878 4.1.1).
bool fse_build(FseTable* t, const int16_t* norm, uint32_t max_sv,
               uint32_t log) noexcept {
    assert(t != nullptr);
    assert(norm != nullptr);
    if (log == 0 || log > kFseMaxLog) return false;
    const uint32_t size = 1u << log;
    const uint32_t mask = size - 1u;
    t->log = log;

    uint16_t next[kMaxSymbols + 1];
    uint32_t high = size - 1u;
    for (uint32_t s = 0; s <= max_sv; ++s) {
        if (norm[s] == -1) {                    // low-prob: seated at the top
            t->symbol[high--] = static_cast<uint8_t>(s);
            next[s] = 1;
        } else {
            next[s] = static_cast<uint16_t>(norm[s]);
        }
    }

    const uint32_t step = (size >> 1) + (size >> 3) + 3u;
    uint32_t pos = 0;
    for (uint32_t s = 0; s <= max_sv; ++s) {
        if (norm[s] <= 0) continue;
        for (int16_t i = 0; i < norm[s]; ++i) {
            t->symbol[pos] = static_cast<uint8_t>(s);
            pos = (pos + step) & mask;
            // Skip cells already owned by low-probability symbols.
            for (uint32_t guard = 0; pos > high; ++guard) {
                if (guard > size) return false;
                pos = (pos + step) & mask;
            }
        }
    }
    if (pos != 0) return false;                 // spread must return to start

    for (uint32_t u = 0; u < size; ++u) {
        const uint8_t  s  = t->symbol[u];
        const uint16_t ns = next[s]++;
        if (ns == 0) return false;
        const uint32_t nb = log - highbit32(ns);
        if (nb > log) return false;
        t->nbits[u]    = static_cast<uint8_t>(nb);
        t->newstate[u] = static_cast<uint16_t>((static_cast<uint32_t>(ns) << nb) - size);
        if (t->newstate[u] >= size) return false;
    }
    return true;
}

bool fse_build_rle(FseTable* t, uint8_t sym) noexcept {
    assert(t != nullptr);
    t->log = 0;
    t->symbol[0] = sym;
    t->nbits[0] = 0;
    t->newstate[0] = 0;
    return true;
}

inline uint32_t fse_init_state(const FseTable* t, ZBits* b) noexcept {
    return zb_read(b, t->log);
}

inline uint32_t fse_next_state(const FseTable* t, uint32_t state, ZBits* b) noexcept {
    assert(state < (1u << t->log) || t->log == 0);
    const uint32_t nb = t->nbits[state];
    return static_cast<uint32_t>(t->newstate[state]) + zb_read(b, nb);
}

// ---------------------------------------------------------------------------
// 3. Huffman
// ---------------------------------------------------------------------------
struct HufTable {
    uint8_t symbol[kHufTableSize];
    uint8_t nbits[kHufTableSize];
    uint32_t maxbits;
    bool     valid;
};

// Build the Huffman decode table from per-symbol weights (RFC 8878 4.2.1).
bool huf_build(HufTable* h, const uint8_t* weights, uint32_t nweights) noexcept {
    assert(h != nullptr);
    assert(weights != nullptr);
    if (nweights == 0 || nweights >= kMaxSymbols) return false;

    uint32_t total = 0;
    uint32_t maxw  = 0;
    for (uint32_t i = 0; i < nweights; ++i) {
        const uint8_t w = weights[i];
        if (w > kHufMaxBits) return false;
        if (w > maxw) maxw = w;
        if (w > 0) total += 1u << (w - 1u);
    }
    if (total == 0) return false;
    const uint32_t maxbits = highbit32(total) + 1u;
    if (maxbits > kHufMaxBits) return false;
    const uint32_t size = 1u << maxbits;
    const uint32_t rest = size - total;
    if (rest == 0 || (rest & (rest - 1u)) != 0) return false;  // must be 2^k
    const uint32_t lastw = highbit32(rest) + 1u;
    if (lastw > maxbits) return false;

    uint8_t wl[kMaxSymbols];
    for (uint32_t i = 0; i < nweights; ++i) wl[i] = weights[i];
    wl[nweights] = static_cast<uint8_t>(lastw);
    const uint32_t nsym = nweights + 1u;

    uint32_t rank_count[kHufMaxBits + 2];
    for (uint32_t i = 0; i < kHufMaxBits + 2; ++i) rank_count[i] = 0;
    for (uint32_t i = 0; i < nsym; ++i) rank_count[wl[i]]++;

    // Weight-1 symbols (longest codes) occupy the lowest table indices.
    uint32_t rank_start[kHufMaxBits + 2];
    for (uint32_t i = 0; i < kHufMaxBits + 2; ++i) rank_start[i] = 0;
    uint32_t acc = 0;
    for (uint32_t w = 1; w <= maxbits; ++w) {
        const uint32_t cur = acc;
        acc += rank_count[w] << (w - 1u);
        rank_start[w] = cur;
    }
    if (acc != size) return false;

    for (uint32_t i = 0; i < nsym; ++i) {
        const uint32_t w = wl[i];
        if (w == 0) continue;
        const uint32_t len = 1u << (w - 1u);
        const uint32_t nb  = maxbits + 1u - w;
        uint32_t st = rank_start[w];
        if (st + len > size) return false;
        for (uint32_t u = 0; u < len; ++u) {
            h->symbol[st + u] = static_cast<uint8_t>(i);
            h->nbits[st + u]  = static_cast<uint8_t>(nb);
        }
        rank_start[w] = st + len;
    }
    h->maxbits = maxbits;
    h->valid   = true;
    return true;
}

// Decode the Huffman tree description. Returns bytes consumed, 0 on error.
uint32_t huf_read_tree(HufTable* h, const uint8_t* src, uint64_t src_len) noexcept {
    assert(h != nullptr);
    if (src_len < 1) return 0;
    const uint8_t hb = src[0];
    uint8_t weights[kMaxSymbols];

    if (hb >= 128) {                       // direct: 4-bit weights, packed
        const uint32_t n = static_cast<uint32_t>(hb) - 127u;
        if (n >= kMaxSymbols) return 0;
        const uint32_t bytes = (n + 1u) / 2u;
        if (1u + bytes > src_len) return 0;
        for (uint32_t i = 0; i < n; ++i) {
            const uint8_t byte = src[1u + (i >> 1)];
            weights[i] = (i & 1u) ? (byte & 0x0Fu) : (byte >> 4);
        }
        if (!huf_build(h, weights, n)) return 0;
        return 1u + bytes;
    }

    // FSE-coded weights: `hb` is the compressed size that follows.
    const uint32_t csize = hb;
    if (csize == 0 || 1u + csize > src_len) return 0;
    int16_t norm[kMaxSymbols];
    uint32_t log = 0;
    const uint32_t hdr = fse_read_ncount(src + 1, csize, norm, kHufMaxBits,
                                         &log, kWeightLog);
    if (hdr == 0 || hdr > csize) return 0;
    FseTable ft;
    if (!fse_build(&ft, norm, kHufMaxBits, log)) return 0;
    if (log > kWeightLog) return 0;
    (void)kWeightTblSize;

    ZBits b;
    if (!zb_init(&b, src + 1 + hdr, csize - hdr)) return 0;
    uint32_t s1 = fse_init_state(&ft, &b);
    uint32_t s2 = fse_init_state(&ft, &b);
    if (zb_overrun(&b)) return 0;

    // Two interleaved states (state1 = even symbols, state2 = odd), decoded
    // until the stream runs out. Per RFC 8878: update the state FIRST, then
    // test for overrun -- an overrun leaves one symbol still live in the OTHER
    // state, which is emitted before stopping.
    const uint32_t tsize = 1u << ft.log;
    uint32_t n = 0;
    for (uint32_t guard = 0; ; ++guard) {
        if (guard > kMaxSymbols + 2u) return 0;
        // Up to THREE weights can be written per iteration (one per state plus
        // the trailing one emitted on the overrun break), so three must fit.
        if (n + 3u > kMaxSymbols) return 0;
        if (s1 >= tsize || s2 >= tsize) return 0;
        weights[n++] = ft.symbol[s1];
        s1 = fse_next_state(&ft, s1, &b);
        if (zb_overrun(&b)) { weights[n++] = ft.symbol[s2]; break; }
        if (s1 >= tsize || s2 >= tsize) return 0;
        weights[n++] = ft.symbol[s2];
        s2 = fse_next_state(&ft, s2, &b);
        if (zb_overrun(&b)) { weights[n++] = ft.symbol[s1]; break; }
    }
    if (!huf_build(h, weights, n)) return 0;
    return 1u + csize;
}

// Decode `count` symbols from one Huffman bitstream.
bool huf_decode_stream(const HufTable* h, const uint8_t* src, uint64_t len,
                       uint8_t* dst, uint64_t count) noexcept {
    assert(h != nullptr);
    assert(h->valid);
    if (count == 0) return true;          // empty stream contributes nothing
    ZBits b;
    if (!zb_init(&b, src, len)) return false;
    const uint32_t mb = h->maxbits;
    for (uint64_t i = 0; i < count; ++i) {
        const uint32_t idx = zb_peek(&b, mb);
        const uint8_t  nb  = h->nbits[idx];
        dst[i] = h->symbol[idx];
        zb_skip(&b, nb);
        if (zb_overrun(&b)) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// 4. Literals section
// ---------------------------------------------------------------------------
struct LitResult {
    const uint8_t* ptr;
    uint64_t       len;
    uint32_t       consumed;   // bytes of block body used
};

bool parse_lit_header(const uint8_t* b, uint64_t n, uint32_t* type,
                      uint32_t* regen, uint32_t* comp, uint32_t* hdr,
                      uint32_t* streams) noexcept {
    if (n < 1) return false;
    const uint32_t t  = b[0] & 3u;
    const uint32_t sf = (b[0] >> 2) & 3u;
    *type = t;
    if (t == 0 || t == 1) {                       // Raw / RLE
        if ((sf & 1u) == 0) {
            *hdr = 1; *regen = b[0] >> 3;
        } else if (sf == 1) {
            if (n < 2) return false;
            *hdr = 2; *regen = (b[0] >> 4) | (static_cast<uint32_t>(b[1]) << 4);
        } else {
            if (n < 3) return false;
            *hdr = 3; *regen = (b[0] >> 4) | (static_cast<uint32_t>(b[1]) << 4) |
                               (static_cast<uint32_t>(b[2]) << 12);
        }
        *comp = 0; *streams = 0;
        return true;
    }
    // Compressed / Treeless
    if (sf == 0 || sf == 1) {
        if (n < 3) return false;
        const uint32_t v = b[0] | (static_cast<uint32_t>(b[1]) << 8) |
                           (static_cast<uint32_t>(b[2]) << 16);
        *hdr = 3; *regen = (v >> 4) & 0x3FFu; *comp = (v >> 14) & 0x3FFu;
        *streams = (sf == 0) ? 1u : 4u;
    } else if (sf == 2) {
        if (n < 4) return false;
        const uint32_t v = b[0] | (static_cast<uint32_t>(b[1]) << 8) |
                           (static_cast<uint32_t>(b[2]) << 16) |
                           (static_cast<uint32_t>(b[3]) << 24);
        *hdr = 4; *regen = (v >> 4) & 0x3FFFu; *comp = (v >> 18) & 0x3FFFu;
        *streams = 4;
    } else {
        if (n < 5) return false;
        const uint64_t v = b[0] | (static_cast<uint64_t>(b[1]) << 8) |
                           (static_cast<uint64_t>(b[2]) << 16) |
                           (static_cast<uint64_t>(b[3]) << 24) |
                           (static_cast<uint64_t>(b[4]) << 32);
        *hdr = 5;
        *regen = static_cast<uint32_t>((v >> 4) & 0x3FFFFu);
        *comp  = static_cast<uint32_t>((v >> 22) & 0x3FFFFu);
        *streams = 4;
    }
    return true;
}

bool decode_lit_huff(HufTable* h, const uint8_t* body, uint64_t blen,
                     uint32_t type, uint32_t regen, uint32_t comp,
                     uint32_t hdr, uint32_t streams,
                     uint8_t* lit, LitResult* out) noexcept {
    if (comp == 0 || static_cast<uint64_t>(hdr) + comp > blen) return false;
    const uint8_t* q  = body + hdr;
    uint64_t       qn = comp;
    if (type == 2) {                              // Compressed: tree first
        const uint32_t used = huf_read_tree(h, q, qn);
        if (used == 0 || used > qn) return false;
        q += used; qn -= used;
    } else {                                      // Treeless: reuse prior tree
        if (!h->valid) return false;
    }
    if (streams == 1) {
        if (!huf_decode_stream(h, q, qn, lit, regen)) return false;
    } else {
        if (qn < 6) return false;
        const uint32_t s1 = static_cast<uint32_t>(q[0]) | (static_cast<uint32_t>(q[1]) << 8);
        const uint32_t s2 = static_cast<uint32_t>(q[2]) | (static_cast<uint32_t>(q[3]) << 8);
        const uint32_t s3 = static_cast<uint32_t>(q[4]) | (static_cast<uint32_t>(q[5]) << 8);
        const uint64_t tot = qn - 6u;
        if (static_cast<uint64_t>(s1) + s2 + s3 > tot) return false;
        const uint64_t s4 = tot - s1 - s2 - s3;
        const uint64_t seg = (regen + 3u) / 4u;
        if (seg * 3u > regen) return false;
        const uint64_t last = regen - seg * 3u;
        const uint8_t* p0 = q + 6;
        if (!huf_decode_stream(h, p0,                s1, lit,            seg))  return false;
        if (!huf_decode_stream(h, p0 + s1,           s2, lit + seg,      seg))  return false;
        if (!huf_decode_stream(h, p0 + s1 + s2,      s3, lit + seg * 2u, seg))  return false;
        if (!huf_decode_stream(h, p0 + s1 + s2 + s3, s4, lit + seg * 3u, last)) return false;
    }
    out->ptr = lit; out->len = regen; out->consumed = hdr + comp;
    return true;
}

bool decode_literals(HufTable* h, const uint8_t* body, uint64_t blen,
                     uint8_t* lit, LitResult* out) noexcept {
    assert(out != nullptr);
    uint32_t type = 0, regen = 0, comp = 0, hdr = 0, streams = 0;
    if (!parse_lit_header(body, blen, &type, &regen, &comp, &hdr, &streams)) {
        return false;
    }
    if (regen > kBlockMax) return false;
    if (type == 0) {                                   // Raw
        if (static_cast<uint64_t>(hdr) + regen > blen) return false;
        out->ptr = body + hdr; out->len = regen; out->consumed = hdr + regen;
        return true;
    }
    if (type == 1) {                                   // RLE
        if (static_cast<uint64_t>(hdr) + 1u > blen) return false;
        memset(lit, body[hdr], regen);
        out->ptr = lit; out->len = regen; out->consumed = hdr + 1u;
        return true;
    }
    return decode_lit_huff(h, body, blen, type, regen, comp, hdr, streams,
                           lit, out);
}

// ---------------------------------------------------------------------------
// 5. Sequences
// ---------------------------------------------------------------------------
const int16_t kLLDefault[36] = {
     4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1,
     2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1,
    -1,-1,-1,-1 };
// NOTE the SEVEN trailing -1 entries (symbols 46..52), not five: symbols 46
// and 47 are low-probability too. Getting this wrong shifts every
// low-probability symbol's table slot and silently mis-decodes long matches
// (it made a 100000-byte RLE block decode to 38 bytes). Positive counts sum to
// 57; 57 + 7 low-prob == 64 == 1<<6, which is the invariant to check against.
const int16_t kMLDefault[53] = {
     1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,
    -1,-1,-1,-1,-1 };
const int16_t kOFDefault[29] = {
     1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, -1,-1,-1,-1,-1 };

// Largest symbol each PREDEFINED distribution defines. Deliberately distinct
// from the kXXSymbolMax caps above, which bound FSE_Compressed tables.
constexpr uint32_t kLLDefMaxSym = 35;   // kLLDefault has 36 entries
constexpr uint32_t kMLDefMaxSym = 52;   // kMLDefault has 53 entries
constexpr uint32_t kOFDefMaxSym = 28;   // kOFDefault has 29 entries (not 32)

const uint32_t kLLBase[36] = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
    16,18,20,22,24,28,32,40,48,64,128,256,512,1024,2048,4096,
    8192,16384,32768,65536 };
const uint8_t kLLBits[36] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,2,2,3,3,4,6,7,8,9,10,11,12,
    13,14,15,16 };
const uint32_t kMLBase[53] = {
    3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,
    19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,
    35,37,39,41,43,47,51,59,67,83,99,131,259,515,1027,2051,
    4099,8195,16387,32771,65539 };
const uint8_t kMLBits[53] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,2,2,3,3,4,4,5,7,8,9,10,11,
    12,13,14,15,16 };

// Build one of the three sequence FSE tables per its compression mode.
// Returns bytes consumed from `p`, or -1 on error.
//
// `max_sv` is the largest symbol an FSE_Compressed table may define; the
// PREDEFINED distribution can be shorter (offsets: 0..28 predefined vs 0..31
// compressed), so it carries its own `def_max_sv`. Conflating the two reads
// past the end of the default table and builds a corrupt one.
int64_t build_seq_table(FseTable* t, uint32_t mode, const uint8_t* p,
                        uint64_t n, uint32_t max_sv, const int16_t* def,
                        uint32_t def_max_sv, uint32_t def_log, uint32_t max_log,
                        bool* have_prev) noexcept {
    assert(t != nullptr);
    assert(have_prev != nullptr);
    assert(def_max_sv <= max_sv);
    switch (mode) {
        case 0:                                        // Predefined
            if (!fse_build(t, def, def_max_sv, def_log)) return -1;
            *have_prev = true;
            return 0;
        case 1: {                                      // RLE
            if (n < 1) return -1;
            if (p[0] > max_sv) return -1;
            if (!fse_build_rle(t, p[0])) return -1;
            *have_prev = true;
            return 1;
        }
        case 2: {                                      // FSE_Compressed
            int16_t norm[kMaxSymbols];
            uint32_t log = 0;
            const uint32_t used = fse_read_ncount(p, n, norm, max_sv, &log,
                                                  max_log);
            if (used == 0) return -1;
            if (!fse_build(t, norm, max_sv, log)) return -1;
            *have_prev = true;
            return static_cast<int64_t>(used);
        }
        case 3:                                        // Repeat previous
            if (!*have_prev) return -1;
            return 0;
        default:
            return -1;
    }
}

struct SeqState {
    FseTable ll, of, ml;
    bool     have_ll, have_of, have_ml;
};

// Apply zstd's repeat-offset rules. Returns the literal match offset.
uint32_t resolve_offset(uint32_t of_value, uint32_t lit_len,
                        uint32_t* rep) noexcept {
    assert(rep != nullptr);
    if (of_value > 3u) {
        rep[2] = rep[1]; rep[1] = rep[0];
        rep[0] = of_value - 3u;
        return rep[0];
    }
    uint32_t idx = of_value + (lit_len == 0 ? 1u : 0u);   // 1..4
    if (idx == 1u) return rep[0];
    uint32_t cand = (idx == 4u) ? (rep[0] - 1u) : rep[idx - 1u];
    if (cand == 0) cand = 1u;
    if (idx != 2u) rep[2] = rep[1];
    rep[1] = rep[0];
    rep[0] = cand;
    return cand;
}

// Copy one sequence's literals + match into dst. Bounds-checked.
bool exec_sequence(uint8_t* dst, uint64_t cap, uint64_t* wr,
                   const uint8_t** lit, uint64_t* lit_left,
                   uint32_t lit_len, uint32_t match_len,
                   uint32_t offset) noexcept {
    assert(wr != nullptr);
    if (lit_len > *lit_left) return false;
    if (*wr + lit_len > cap) return false;
    memcpy(dst + *wr, *lit, lit_len);
    *wr += lit_len; *lit += lit_len; *lit_left -= lit_len;

    if (offset == 0) return false;
    if (static_cast<uint64_t>(offset) > *wr) return false;   // before output
    if (*wr + match_len > cap) return false;
    uint64_t from = *wr - offset;
    for (uint32_t i = 0; i < match_len; ++i) {   // byte-wise: may overlap
        dst[*wr + i] = dst[from + i];
    }
    *wr += match_len;
    return true;
}

// Decode + execute the sequences section of a compressed block.
// Read Number_of_Sequences + the symbol-compression-modes byte, and build the
// three FSE tables. Returns the offset of the sequence bitstream, or -1.
int64_t read_seq_tables(SeqState* st, const uint8_t* p, uint64_t n,
                        uint32_t* out_nseq) noexcept {
    assert(st != nullptr);
    assert(out_nseq != nullptr);
    if (n < 1) return -1;
    uint64_t o = 0;
    uint32_t nseq = p[o++];
    if (nseq >= 128) {
        if (nseq < 255) {
            if (o >= n) return -1;
            nseq = ((nseq - 128u) << 8) + p[o++];
        } else {
            if (o + 2 > n) return -1;
            nseq = static_cast<uint32_t>(p[o]) +
                   (static_cast<uint32_t>(p[o + 1]) << 8) + 0x7F00u;
            o += 2;
        }
    }
    *out_nseq = nseq;
    if (nseq == 0) return static_cast<int64_t>(o);
    if (o >= n) return -1;
    const uint8_t scm = p[o++];
    if ((scm & 3u) != 0) return -1;                  // reserved bits must be 0

    const uint32_t modes[3] = {(scm >> 6) & 3u, (scm >> 4) & 3u,
                               (scm >> 2) & 3u};     // LL, OF, ML -- in order
    FseTable* tabs[3] = {&st->ll, &st->of, &st->ml};
    bool* haves[3] = {&st->have_ll, &st->have_of, &st->have_ml};
    const int16_t* defs[3] = {kLLDefault, kOFDefault, kMLDefault};
    const uint32_t maxsv[3] = {kLLSymbolMax, kOFSymbolMax, kMLSymbolMax};
    const uint32_t dmax[3] = {kLLDefMaxSym, kOFDefMaxSym, kMLDefMaxSym};
    const uint32_t dlog[3] = {6, 5, 6};
    const uint32_t mlog[3] = {kLLMaxLog, kOFMaxLog, kMLMaxLog};
    for (uint32_t i = 0; i < 3; ++i) {
        const int64_t u = build_seq_table(tabs[i], modes[i], p + o, n - o,
                                          maxsv[i], defs[i], dmax[i], dlog[i],
                                          mlog[i], haves[i]);
        if (u < 0) return -1;
        o += static_cast<uint64_t>(u);
        if (o > n) return -1;
    }
    return static_cast<int64_t>(o);
}

// Decode `nseq` sequences from the backward bitstream and execute them.
int run_sequences(SeqState* st, uint32_t nseq, const uint8_t* p, uint64_t n,
                  const uint8_t* lit, uint64_t lit_len,
                  uint8_t* dst, uint64_t cap, uint64_t* wr,
                  uint32_t* rep) noexcept {
    assert(st != nullptr);
    assert(wr != nullptr && rep != nullptr);
    ZBits b;
    if (!zb_init(&b, p, n)) return kZstdBadInput;
    uint32_t s_ll = fse_init_state(&st->ll, &b);
    uint32_t s_of = fse_init_state(&st->of, &b);
    uint32_t s_ml = fse_init_state(&st->ml, &b);
    if (zb_overrun(&b)) return kZstdBadInput;

    const uint8_t* lp = lit;
    uint64_t       ll_left = lit_len;
    for (uint32_t i = 0; i < nseq; ++i) {
        const uint8_t ll_code = st->ll.symbol[s_ll];
        const uint8_t of_code = st->of.symbol[s_of];
        const uint8_t ml_code = st->ml.symbol[s_ml];
        // of_code is the offset's extra-bit count; > 31 would shift out of
        // range, so this bound is load-bearing, not defensive noise.
        if (ll_code > kLLSymbolMax || ml_code > kMLSymbolMax ||
            of_code > kOFSymbolMax) {
            return kZstdBadInput;
        }
        // Value bits are read offset-first, then match, then literal length.
        const uint32_t of_val = (1u << of_code) + zb_read(&b, of_code);
        const uint32_t ml = kMLBase[ml_code] + zb_read(&b, kMLBits[ml_code]);
        const uint32_t ll = kLLBase[ll_code] + zb_read(&b, kLLBits[ll_code]);
        if (zb_overrun(&b)) return kZstdBadInput;
        const uint32_t off = resolve_offset(of_val, ll, rep);
        if (!exec_sequence(dst, cap, wr, &lp, &ll_left, ll, ml, off)) {
            return (*wr + ll + ml > cap) ? kZstdNeedMore : kZstdBadInput;
        }
        if (i + 1u < nseq) {          // states update only between sequences
            s_ll = fse_next_state(&st->ll, s_ll, &b);
            s_ml = fse_next_state(&st->ml, s_ml, &b);
            s_of = fse_next_state(&st->of, s_of, &b);
            if (zb_overrun(&b)) return kZstdBadInput;
        }
    }
    if (*wr + ll_left > cap) return kZstdNeedMore;
    memcpy(dst + *wr, lp, ll_left);          // trailing literals
    *wr += ll_left;
    return kZstdOk;
}

int decode_sequences(SeqState* st, const uint8_t* p, uint64_t n,
                     const uint8_t* lit, uint64_t lit_len,
                     uint8_t* dst, uint64_t cap, uint64_t* wr,
                     uint32_t* rep) noexcept {
    assert(st != nullptr);
    assert(wr != nullptr);
    uint32_t nseq = 0;
    const int64_t off = read_seq_tables(st, p, n, &nseq);
    if (off < 0) return kZstdBadInput;
    const uint64_t o = static_cast<uint64_t>(off);
    if (nseq == 0) {                                 // no sequences: literals
        if (*wr + lit_len > cap) return kZstdNeedMore;
        memcpy(dst + *wr, lit, lit_len);
        *wr += lit_len;
        return (o == n) ? kZstdOk : kZstdBadInput;
    }
    if (o >= n) return kZstdBadInput;
    return run_sequences(st, nseq, p + o, n - o, lit, lit_len, dst, cap, wr,
                         rep);
}

// ---------------------------------------------------------------------------
// 7a. xxhash64 — used only to verify the optional frame content checksum.
// ---------------------------------------------------------------------------
constexpr uint64_t kP1 = 11400714785074694791ull;
constexpr uint64_t kP2 = 14029467366897019727ull;
constexpr uint64_t kP3 =  1609587929392839161ull;
constexpr uint64_t kP4 =  9650029242287828579ull;
constexpr uint64_t kP5 =  2870177450012600261ull;

inline uint64_t rotl64(uint64_t v, int r) noexcept {
    return (v << r) | (v >> (64 - r));
}
inline uint64_t rd64(const uint8_t* p) noexcept {
    uint64_t v; memcpy(&v, p, 8); return v;
}
inline uint64_t rd32(const uint8_t* p) noexcept {
    uint32_t v; memcpy(&v, p, 4); return v;
}
inline uint64_t xxh_round(uint64_t acc, uint64_t in) noexcept {
    acc += in * kP2; acc = rotl64(acc, 31); acc *= kP1; return acc;
}
inline uint64_t xxh_merge(uint64_t acc, uint64_t val) noexcept {
    acc ^= xxh_round(0, val); acc = acc * kP1 + kP4; return acc;
}

uint64_t xxhash64(const uint8_t* p, uint64_t len) noexcept {
    const uint8_t* const end = p + len;
    uint64_t h;
    if (len >= 32) {
        uint64_t v1 = kP1 + kP2, v2 = kP2, v3 = 0, v4 = 0ull - kP1;
        const uint8_t* const lim = end - 32;
        do {
            v1 = xxh_round(v1, rd64(p)); p += 8;
            v2 = xxh_round(v2, rd64(p)); p += 8;
            v3 = xxh_round(v3, rd64(p)); p += 8;
            v4 = xxh_round(v4, rd64(p)); p += 8;
        } while (p <= lim);
        h = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);
        h = xxh_merge(h, v1); h = xxh_merge(h, v2);
        h = xxh_merge(h, v3); h = xxh_merge(h, v4);
    } else {
        h = kP5;
    }
    h += len;
    while (p + 8 <= end) {
        h ^= xxh_round(0, rd64(p));
        h = rotl64(h, 27) * kP1 + kP4;
        p += 8;
    }
    if (p + 4 <= end) {
        h ^= rd32(p) * kP1;
        h = rotl64(h, 23) * kP2 + kP3;
        p += 4;
    }
    while (p < end) {
        h ^= static_cast<uint64_t>(*p) * kP5;
        h = rotl64(h, 11) * kP1;
        ++p;
    }
    h ^= h >> 33; h *= kP2;
    h ^= h >> 29; h *= kP3;
    h ^= h >> 32;
    return h;
}

// ---------------------------------------------------------------------------
// 6/7b. Scratch, blocks, frames
// ---------------------------------------------------------------------------
struct Scratch {
    uint8_t  lit[kBlockMax];
    HufTable huf;
    SeqState seq;
};

// Decode one compressed block body into dst.
int decode_compressed_block(Scratch* sc, const uint8_t* body, uint64_t blen,
                            uint8_t* dst, uint64_t cap, uint64_t* wr,
                            uint32_t* rep) noexcept {
    assert(sc != nullptr);
    LitResult lr{nullptr, 0, 0};
    if (!decode_literals(&sc->huf, body, blen, sc->lit, &lr)) {
        return kZstdBadInput;
    }
    if (lr.consumed > blen) return kZstdBadInput;
    return decode_sequences(&sc->seq, body + lr.consumed, blen - lr.consumed,
                            lr.ptr, lr.len, dst, cap, wr, rep);
}

// Parse a frame header. Returns bytes consumed, or 0 on error.
uint32_t parse_frame_header(const uint8_t* p, uint64_t n, bool* checksum,
                            uint64_t* fcs_out, bool* have_fcs,
                            int* err) noexcept {
    assert(checksum != nullptr);
    assert(err != nullptr);
    assert(fcs_out != nullptr && have_fcs != nullptr);
    *err = kZstdOk;
    *have_fcs = false;
    *fcs_out = 0;
    if (n < 5) { *err = kZstdTruncSrc; return 0; }
    uint64_t o = 4;                                  // magic already matched
    const uint8_t fhd = p[o++];
    const uint32_t fcs_flag   = (fhd >> 6) & 3u;
    const uint32_t single_seg = (fhd >> 5) & 1u;
    const uint32_t reserved   = (fhd >> 3) & 1u;
    *checksum = ((fhd >> 2) & 1u) != 0;
    const uint32_t did_flag   = fhd & 3u;
    if (reserved != 0) { *err = kZstdBadInput; return 0; }
    if (!single_seg) {
        if (o >= n) { *err = kZstdTruncSrc; return 0; }
        ++o;                                         // window descriptor
    }
    static const uint32_t kDidLen[4] = {0, 1, 2, 4};
    const uint32_t did = kDidLen[did_flag];
    if (o + did > n) { *err = kZstdTruncSrc; return 0; }
    if (did != 0) {
        // A dictionary frame needs a dictionary we have no API to supply.
        // Refuse loudly rather than decode to wrong bytes.
        *err = kZstdUnsupported;
        return 0;
    }
    const uint32_t kFcsLen[4] = {single_seg ? 1u : 0u, 2u, 4u, 8u};
    const uint32_t fcs = kFcsLen[fcs_flag];
    if (o + fcs > n) { *err = kZstdTruncSrc; return 0; }
    if (fcs != 0) {
        uint64_t v = 0;
        for (uint32_t i = 0; i < fcs; ++i) {
            v |= static_cast<uint64_t>(p[o + i]) << (8u * i);
        }
        if (fcs == 2) v += 256u;          // the 2-byte form is biased
        *fcs_out = v;
        *have_fcs = true;
    }
    o += fcs;
    return static_cast<uint32_t>(o);
}

// Validate a frame's declared size and optional content checksum, consuming
// the 4 checksum bytes from *o when present.
int check_frame_trailer(const uint8_t* p, uint64_t n, uint64_t* o,
                        const uint8_t* dst, uint64_t frame_start, uint64_t wr,
                        bool have_fcs, uint64_t fcs, bool checksum) noexcept {
    assert(o != nullptr);
    assert(wr >= frame_start);
    // When the writer declared the frame's decompressed size, hold it to that.
    // A frame that decodes SHORT is corrupt, and without this check it would
    // be returned as a silently truncated result.
    if (have_fcs && (wr - frame_start) != fcs) return kZstdBadInput;
    if (!checksum) return kZstdOk;
    if (*o + 4 > n) return kZstdTruncSrc;
    const uint32_t want = static_cast<uint32_t>(p[*o]) |
                          (static_cast<uint32_t>(p[*o + 1]) << 8) |
                          (static_cast<uint32_t>(p[*o + 2]) << 16) |
                          (static_cast<uint32_t>(p[*o + 3]) << 24);
    *o += 4;
    const uint64_t got = xxhash64(dst + frame_start, wr - frame_start);
    return (static_cast<uint32_t>(got & 0xFFFFFFFFull) == want)
         ? kZstdOk : kZstdChecksum;
}

// Decode one frame starting at p[0]. *consumed gets the frame's byte length.
int decode_frame(Scratch* sc, const uint8_t* p, uint64_t n,
                 uint8_t* dst, uint64_t cap, uint64_t* wr,
                 uint64_t* consumed) noexcept {
    assert(sc != nullptr);
    assert(consumed != nullptr);
    bool     checksum = false;
    bool     have_fcs = false;
    uint64_t fcs = 0;
    int      err = kZstdOk;
    const uint32_t hdr = parse_frame_header(p, n, &checksum, &fcs, &have_fcs,
                                            &err);
    if (hdr == 0) return (err == kZstdOk) ? kZstdBadInput : err;

    sc->huf.valid = false;                 // Huffman/FSE reuse is frame-scoped
    sc->seq.have_ll = sc->seq.have_of = sc->seq.have_ml = false;
    uint32_t rep[3] = {1, 4, 8};
    const uint64_t frame_start = *wr;

    uint64_t o = hdr;
    for (uint32_t blk = 0; ; ++blk) {
        if (blk >= kMaxBlocksPerFrame) return kZstdBadInput;
        if (o + 3 > n) return kZstdTruncSrc;
        const uint32_t bh = static_cast<uint32_t>(p[o]) |
                            (static_cast<uint32_t>(p[o + 1]) << 8) |
                            (static_cast<uint32_t>(p[o + 2]) << 16);
        o += 3;
        const uint32_t last  = bh & 1u;
        const uint32_t btype = (bh >> 1) & 3u;
        const uint32_t bsize = bh >> 3;
        if (bsize > kBlockMax) return kZstdBadInput;
        if (btype == 3) return kZstdBadInput;          // reserved
        if (btype == 1) {                              // RLE block
            if (o + 1 > n) return kZstdTruncSrc;
            if (*wr + bsize > cap) return kZstdNeedMore;
            memset(dst + *wr, p[o], bsize);
            *wr += bsize;
            o += 1;
        } else {
            if (o + bsize > n) return kZstdTruncSrc;
            if (btype == 0) {                          // Raw block
                if (*wr + bsize > cap) return kZstdNeedMore;
                memcpy(dst + *wr, p + o, bsize);
                *wr += bsize;
            } else {                                   // Compressed block
                const int r = decode_compressed_block(sc, p + o, bsize, dst,
                                                      cap, wr, rep);
                if (r != kZstdOk) return r;
            }
            o += bsize;
        }
        if (last) break;
    }
    const int tr = check_frame_trailer(p, n, &o, dst, frame_start, *wr,
                                       have_fcs, fcs, checksum);
    if (tr != kZstdOk) return tr;
    *consumed = o;
    return kZstdOk;
}

// A skippable frame carries no data; step over it.
bool skippable_len(const uint8_t* p, uint64_t n, uint64_t* out) noexcept {
    assert(out != nullptr);
    if (n < 8) return false;
    const uint32_t sz = static_cast<uint32_t>(p[4]) |
                        (static_cast<uint32_t>(p[5]) << 8) |
                        (static_cast<uint32_t>(p[6]) << 16) |
                        (static_cast<uint32_t>(p[7]) << 24);
    if (static_cast<uint64_t>(sz) + 8u > n) return false;
    *out = static_cast<uint64_t>(sz) + 8u;
    return true;
}

}  // namespace

uint64_t zstd_scratch_size() noexcept {
    return sizeof(Scratch);
}

int zstd_decode_raw(const uint8_t* src, uint64_t src_len,
                    uint8_t* dst, uint64_t dst_cap, uint64_t* out_len,
                    void* scratch, uint64_t scratch_len) noexcept {
    assert(out_len != nullptr);
    assert(src != nullptr || src_len == 0);
    if (out_len == nullptr) return kZstdBadInput;
    *out_len = 0;
    if (src == nullptr || src_len == 0) return kZstdTruncSrc;
    if (dst == nullptr && dst_cap != 0) return kZstdBadInput;
    if (scratch == nullptr || scratch_len < sizeof(Scratch)) {
        return kZstdScratchSmall;
    }
    if ((reinterpret_cast<uintptr_t>(scratch) & 7u) != 0) return kZstdBadInput;

    Scratch* sc = static_cast<Scratch*>(scratch);
    sc->huf.valid = false;

    uint64_t o  = 0;
    uint64_t wr = 0;
    for (uint32_t f = 0; o < src_len; ++f) {
        if (f >= kMaxFrames) return kZstdBadInput;
        if (o + 4 > src_len) return kZstdTruncSrc;
        const uint32_t magic = static_cast<uint32_t>(src[o]) |
                               (static_cast<uint32_t>(src[o + 1]) << 8) |
                               (static_cast<uint32_t>(src[o + 2]) << 16) |
                               (static_cast<uint32_t>(src[o + 3]) << 24);
        if ((magic & 0xFFFFFFF0u) == 0x184D2A50u) {     // skippable frame
            uint64_t sk = 0;
            if (!skippable_len(src + o, src_len - o, &sk)) return kZstdTruncSrc;
            o += sk;
            continue;
        }
        if (magic != 0xFD2FB528u) return kZstdBadInput;
        uint64_t used = 0;
        const int r = decode_frame(sc, src + o, src_len - o, dst, dst_cap,
                                   &wr, &used);
        if (r != kZstdOk) return r;
        if (used == 0 || used > src_len - o) return kZstdBadInput;
        o += used;
    }
    *out_len = wr;
    return kZstdOk;
}

}  // namespace ingest
}  // namespace bolt
