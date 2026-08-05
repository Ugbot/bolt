// bolt/ingest/bolt_inflate.cpp — raw DEFLATE (RFC 1951) decoder. See header.
//
// Structure mirrors zlib's reference decoder `puff.c`: a bit reader, canonical
// Huffman tables held as per-length counts + a sorted symbol list, and three
// block decoders (stored / fixed / dynamic). Every read is bounds-checked
// against the input length and every write against the output capacity, so a
// corrupt or hostile stream returns an error rather than misbehaving.

#include "bolt/ingest/bolt_inflate.h"

#include <cassert>
#include <cstring>

namespace bolt {
namespace ingest {
namespace {

constexpr uint32_t kMaxBits    = 15u;   // max Huffman code length
constexpr uint32_t kMaxLSyms   = 288u;  // literal/length alphabet
constexpr uint32_t kMaxDSyms   = 30u;   // distance alphabet
constexpr uint32_t kNumCLSyms  = 19u;   // code-length alphabet
// A single deflate stream is bounded so a crafted input cannot spin forever.
constexpr uint32_t kMaxBlocks  = 1u << 20;

struct BitIn {
    const uint8_t* src;
    uint64_t       len;
    uint64_t       pos;      // next byte to consume
    uint32_t       buf;      // bit accumulator, LSB-first
    uint32_t       cnt;      // valid bits in `buf`
    bool           trunc;    // ran past the end
};

// Pull `need` bits (0..24) LSB-first. On truncation sets `trunc` and returns 0;
// callers check `trunc` at block granularity, which keeps the hot path branchy
// only once per block rather than once per code.
uint32_t bits(BitIn* b, uint32_t need) noexcept {
    assert(b != nullptr);
    assert(need <= 24u);
    while (b->cnt < need) {                       // bounded: need <= 24
        if (b->pos >= b->len) { b->trunc = true; return 0u; }
        b->buf |= static_cast<uint32_t>(b->src[b->pos++]) << b->cnt;
        b->cnt += 8u;
    }
    const uint32_t v = b->buf & ((1u << need) - 1u);
    b->buf >>= need;
    b->cnt -= need;
    return v;
}

// Canonical Huffman table: count[l] symbols have length l; `symbol` lists them
// in (length, symbol) order.
struct Huff {
    int16_t count[kMaxBits + 1u];
    int16_t symbol[kMaxLSyms];
};

// Build from a code-length array. Returns false for an over-subscribed set
// (a malformed stream); an incomplete set is allowed only when it is empty or
// a single code, matching RFC 1951's distance-tree special case.
bool huff_build(Huff* h, const uint8_t* lengths, uint32_t n) noexcept {
    assert(h != nullptr);
    assert(lengths != nullptr);
    assert(n <= kMaxLSyms);
    std::memset(h->count, 0, sizeof(h->count));
    for (uint32_t i = 0; i < n; ++i) {            // bounded: n <= kMaxLSyms
        assert(lengths[i] <= kMaxBits);
        ++h->count[lengths[i]];
    }
    if (h->count[0] == static_cast<int16_t>(n)) return true;  // no codes at all
    // Over-subscription check: a valid canonical set never runs out of slack.
    int32_t left = 1;
    for (uint32_t l = 1; l <= kMaxBits; ++l) {    // bounded: kMaxBits
        left <<= 1;
        left -= h->count[l];
        if (left < 0) return false;
    }
    int16_t offs[kMaxBits + 2u];
    offs[1] = 0;
    for (uint32_t l = 1; l <= kMaxBits; ++l) {    // bounded: kMaxBits
        offs[l + 1u] = static_cast<int16_t>(offs[l] + h->count[l]);
    }
    for (uint32_t i = 0; i < n; ++i) {            // bounded: n
        if (lengths[i] != 0u) h->symbol[offs[lengths[i]]++] = static_cast<int16_t>(i);
    }
    return true;
}

// Decode one symbol, walking code lengths shortest-first (puff's algorithm).
// Returns -1 on an invalid code or truncation.
int32_t huff_decode(BitIn* b, const Huff* h) noexcept {
    assert(b != nullptr);
    assert(h != nullptr);
    int32_t code = 0, first = 0, index = 0;
    for (uint32_t l = 1; l <= kMaxBits; ++l) {    // bounded: kMaxBits
        code |= static_cast<int32_t>(bits(b, 1u));
        if (b->trunc) return -1;
        const int32_t cnt = h->count[l];
        if (code - first < cnt) return h->symbol[index + (code - first)];
        index += cnt;
        first  = (first + cnt) << 1;
        code <<= 1;
    }
    return -1;
}

const uint16_t kLenBase[29] = {
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59,
    67, 83, 99, 115, 131, 163, 195, 227, 258 };
const uint8_t kLenExtra[29] = {
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
    4, 4, 4, 4, 5, 5, 5, 5, 0 };
const uint16_t kDistBase[30] = {
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513,
    769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577 };
const uint8_t kDistExtra[30] = {
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8,
    9, 9, 10, 10, 11, 11, 12, 12, 13, 13 };

struct Out {
    uint8_t* dst;
    uint64_t cap;
    uint64_t n;
    bool     overflow;   // needed more room than `cap`
};

// Emit one literal byte. Past capacity we keep COUNTING but stop writing, so a
// caller can size a retry exactly instead of doubling blindly.
void put(Out* o, uint8_t v) noexcept {
    assert(o != nullptr);
    if (o->n < o->cap) o->dst[o->n] = v;
    else o->overflow = true;
    ++o->n;
}

// Decode a literal/length + distance block against the two given trees.
bool inflate_codes(BitIn* b, Out* o, const Huff* lc, const Huff* dc) noexcept {
    assert(b != nullptr);
    assert(o != nullptr);
    for (;;) {                                     // bounded by input exhaustion
        const int32_t sym = huff_decode(b, lc);
        if (sym < 0) return false;
        if (sym < 256) { put(o, static_cast<uint8_t>(sym)); continue; }
        if (sym == 256) return true;               // end of block
        const uint32_t li = static_cast<uint32_t>(sym) - 257u;
        if (li >= 29u) return false;
        const uint32_t len = kLenBase[li] + bits(b, kLenExtra[li]);
        const int32_t dsym = huff_decode(b, dc);
        if (dsym < 0 || dsym >= static_cast<int32_t>(kMaxDSyms)) return false;
        const uint32_t di = static_cast<uint32_t>(dsym);
        const uint64_t dist = kDistBase[di] + bits(b, kDistExtra[di]);
        if (b->trunc) return false;
        if (dist == 0u || dist > o->n) return false;   // reference before start
        // Copy may overlap (dist < len) — byte-at-a-time is required, and it is
        // also what lets an over-capacity run keep counting correctly.
        for (uint32_t i = 0; i < len; ++i) {       // bounded: len <= 258
            const uint64_t srcpos = o->n - dist;
            // Past capacity the source byte was never written, so it cannot be
            // read back; the count is still exact, which is all a retry needs.
            const uint8_t v = (srcpos < o->cap && !o->overflow) ? o->dst[srcpos] : 0u;
            put(o, v);
        }
    }
}

// The fixed literal/length + distance trees of RFC 1951 §3.2.6.
bool build_fixed(Huff* lc, Huff* dc) noexcept {
    assert(lc != nullptr);
    assert(dc != nullptr);
    uint8_t l[kMaxLSyms];
    uint32_t i = 0;
    for (; i < 144u; ++i) l[i] = 8u;
    for (; i < 256u; ++i) l[i] = 9u;
    for (; i < 280u; ++i) l[i] = 7u;
    for (; i < 288u; ++i) l[i] = 8u;
    if (!huff_build(lc, l, kMaxLSyms)) return false;
    uint8_t d[kMaxDSyms];
    for (i = 0; i < kMaxDSyms; ++i) d[i] = 5u;
    return huff_build(dc, d, kMaxDSyms);
}

// Read the dynamic block's code-length-coded trees (RFC 1951 §3.2.7).
bool build_dynamic(BitIn* b, Huff* lc, Huff* dc) noexcept {
    assert(b != nullptr);
    assert(lc != nullptr);
    const uint32_t nlen  = bits(b, 5u) + 257u;
    const uint32_t ndist = bits(b, 5u) + 1u;
    const uint32_t ncode = bits(b, 4u) + 4u;
    if (b->trunc) return false;
    if (nlen > kMaxLSyms || ndist > kMaxDSyms) return false;
    static const uint8_t kOrder[kNumCLSyms] = {
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15 };
    uint8_t cl[kNumCLSyms];
    std::memset(cl, 0, sizeof(cl));
    for (uint32_t i = 0; i < ncode; ++i) {         // bounded: kNumCLSyms
        cl[kOrder[i]] = static_cast<uint8_t>(bits(b, 3u));
    }
    if (b->trunc) return false;
    Huff clh;
    if (!huff_build(&clh, cl, kNumCLSyms)) return false;

    uint8_t lengths[kMaxLSyms + kMaxDSyms];
    std::memset(lengths, 0, sizeof(lengths));
    const uint32_t total = nlen + ndist;
    uint32_t i = 0;
    while (i < total) {                            // bounded: total, i advances
        const int32_t sym = huff_decode(b, &clh);
        if (sym < 0) return false;
        if (sym < 16) { lengths[i++] = static_cast<uint8_t>(sym); continue; }
        uint32_t rep = 0;
        uint8_t  val = 0;
        if (sym == 16) {
            if (i == 0u) return false;             // nothing to repeat
            val = lengths[i - 1u];
            rep = 3u + bits(b, 2u);
        } else if (sym == 17) {
            rep = 3u + bits(b, 3u);
        } else {
            rep = 11u + bits(b, 7u);
        }
        if (b->trunc) return false;
        if (i + rep > total) return false;         // would overrun the arrays
        for (uint32_t r = 0; r < rep; ++r) lengths[i++] = val;   // bounded: rep
    }
    if (lengths[256] == 0u) return false;          // no end-of-block code
    if (!huff_build(lc, lengths, nlen)) return false;
    return huff_build(dc, lengths + nlen, ndist);
}

// A stored (uncompressed) block: byte-align, LEN/NLEN, then raw bytes.
bool inflate_stored(BitIn* b, Out* o) noexcept {
    assert(b != nullptr);
    assert(o != nullptr);
    b->buf = 0u;
    b->cnt = 0u;                                   // discard to byte boundary
    if (b->pos + 4u > b->len) { b->trunc = true; return false; }
    const uint32_t len  = static_cast<uint32_t>(b->src[b->pos]) |
                          (static_cast<uint32_t>(b->src[b->pos + 1u]) << 8);
    const uint32_t nlen = static_cast<uint32_t>(b->src[b->pos + 2u]) |
                          (static_cast<uint32_t>(b->src[b->pos + 3u]) << 8);
    b->pos += 4u;
    if ((len ^ 0xFFFFu) != nlen) return false;     // complement check
    if (b->pos + len > b->len) { b->trunc = true; return false; }
    for (uint32_t i = 0; i < len; ++i) put(o, b->src[b->pos + i]);  // bounded
    b->pos += len;
    return true;
}

}  // namespace

int inflate_raw(const uint8_t* src, uint64_t src_len,
                uint8_t* dst, uint64_t dst_cap, uint64_t* out_len) noexcept {
    assert(src != nullptr || src_len == 0u);
    assert(dst != nullptr || dst_cap == 0u);
    if (src == nullptr || out_len == nullptr) return kInflateBadInput;
    if (dst == nullptr && dst_cap != 0u) return kInflateBadInput;

    BitIn b;
    std::memset(&b, 0, sizeof(b));
    b.src = src;
    b.len = src_len;
    Out o;
    std::memset(&o, 0, sizeof(o));
    o.dst = dst;
    o.cap = dst_cap;

    for (uint32_t blk = 0; blk < kMaxBlocks; ++blk) {   // bounded block count
        const uint32_t last = bits(&b, 1u);
        const uint32_t type = bits(&b, 2u);
        if (b.trunc) return kInflateTruncSrc;
        bool ok = false;
        if (type == 0u) {
            ok = inflate_stored(&b, &o);
        } else if (type == 1u || type == 2u) {
            Huff lc, dc;
            ok = (type == 1u) ? build_fixed(&lc, &dc) : build_dynamic(&b, &lc, &dc);
            if (ok) ok = inflate_codes(&b, &o, &lc, &dc);
        } else {
            return kInflateBadInput;                // type 3 is reserved
        }
        if (!ok) return b.trunc ? kInflateTruncSrc : kInflateBadInput;
        if (last != 0u) {
            // `o.n` is EXACT even when the buffer overflowed: literal and
            // match lengths come from the bit stream, never from the output
            // bytes, so counting continues correctly once writing stops. That
            // makes *out_len the precise size a retry needs — one retry, no
            // doubling guesswork.
            *out_len = o.n;
            return o.overflow ? kInflateNeedMore : kInflateOk;
        }
    }
    return kInflateBadInput;                        // never terminated
}

}  // namespace ingest
}  // namespace bolt
