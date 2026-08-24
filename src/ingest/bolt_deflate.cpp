// bolt_deflate.cpp — self-contained DEFLATE compressor + GZIP writer.
// See the header for scope and for why fixed-Huffman only.

#include "bolt/ingest/bolt_deflate.h"

#include <cassert>
#include <cstring>

namespace bolt {
namespace ingest {

namespace {

constexpr std::uint32_t kWindow    = 1u << 15;   // 32 KiB, DEFLATE's maximum
constexpr std::uint32_t kMinMatch  = 3u;
constexpr std::uint32_t kMaxMatch  = 258u;
constexpr std::uint32_t kHashBits  = 15u;
constexpr std::uint32_t kHashSize  = 1u << kHashBits;
// Bound on hash-chain walking per position. DEFLATE quality is a smooth
// function of this; 32 is roughly zlib level 6 and keeps the compressor
// linear on adversarial input (long runs of one byte produce a chain as long
// as the window).
constexpr std::uint32_t kMaxChain  = 32u;
// A STORED block's payload length field is 16 bits.
constexpr std::uint32_t kMaxStored = 65535u;
// "no candidate". Must not be a representable position: the match finder
// stores FULL positions, because truncating them to 16 bits made every
// distance past 65535 wrong -- which showed up as 100 KB of one repeated
// byte compressing to 34 KB instead of a few hundred bytes, since the chain
// pointed at garbage the moment the input outgrew a uint16.
constexpr std::uint32_t kNoPos = 0xFFFFFFFFu;

// ---- bit writer -----------------------------------------------------------
//
// DEFLATE packs bits into bytes LSB-first, but a HUFFMAN CODE is written
// most-significant-bit first. Those two rules together are the single most
// common source of a stream that inflates to garbage, so the code path that
// writes a Huffman symbol reverses its bits and the one that writes an
// extra-bits field does not.
struct BitW {
    std::uint8_t*  dst;
    std::uint64_t  cap;
    std::uint64_t  pos;
    std::uint32_t  acc;      // bits waiting, LSB-first
    std::uint32_t  nbits;
    bool           ok;
};

inline void bw_put(BitW* w, std::uint32_t bits, std::uint32_t n) noexcept {
    assert(n <= 24u);
    if (!w->ok) return;
    w->acc |= (bits & ((n == 32u) ? 0xFFFFFFFFu : ((1u << n) - 1u))) << w->nbits;
    w->nbits += n;
    while (w->nbits >= 8u) {
        if (w->pos >= w->cap) { w->ok = false; return; }
        w->dst[w->pos++] = static_cast<std::uint8_t>(w->acc & 0xFFu);
        w->acc >>= 8;
        w->nbits -= 8u;
    }
}

// Reverse the low `n` bits: a Huffman code is transmitted MSB-first through a
// bit stream that is otherwise LSB-first.
inline std::uint32_t rev_bits(std::uint32_t v, std::uint32_t n) noexcept {
    std::uint32_t r = 0;
    for (std::uint32_t i = 0; i < n; ++i) {
        r = (r << 1) | ((v >> i) & 1u);
    }
    return r;
}

inline void bw_align(BitW* w) noexcept {
    if (!w->ok) return;
    if (w->nbits != 0u) {
        if (w->pos >= w->cap) { w->ok = false; return; }
        w->dst[w->pos++] = static_cast<std::uint8_t>(w->acc & 0xFFu);
        w->acc = 0;
        w->nbits = 0;
    }
}

// ---- fixed Huffman tables (RFC 1951 section 3.2.6) ------------------------
//
// literal/length: 0-143 -> 8 bits from 0x30, 144-255 -> 9 bits from 0x190,
// 256-279 -> 7 bits from 0x00, 280-287 -> 8 bits from 0xC0.
inline void fixed_lit(std::uint32_t sym, std::uint32_t* code,
                      std::uint32_t* len) noexcept {
    assert(sym < 288u);
    if (sym < 144u)      { *code = 0x30u + sym;          *len = 8u; }
    else if (sym < 256u) { *code = 0x190u + (sym - 144u); *len = 9u; }
    else if (sym < 280u) { *code = sym - 256u;            *len = 7u; }
    else                 { *code = 0xC0u + (sym - 280u);  *len = 8u; }
}

inline void bw_sym(BitW* w, std::uint32_t sym) noexcept {
    std::uint32_t code = 0, len = 0;
    fixed_lit(sym, &code, &len);
    bw_put(w, rev_bits(code, len), len);
}

// Distance codes are a flat 5-bit fixed code, also MSB-first.
inline void bw_dist_sym(BitW* w, std::uint32_t sym) noexcept {
    assert(sym < 30u);
    bw_put(w, rev_bits(sym, 5u), 5u);
}

// Length code tables. Index by (length - 3).
struct LenCode { std::uint16_t sym; std::uint8_t extra; std::uint16_t base; };

LenCode len_code_for(std::uint32_t len) noexcept {
    assert(len >= kMinMatch && len <= kMaxMatch);
    static const std::uint16_t kBase[29] = {
        3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43,
        51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
    static const std::uint8_t kExtra[29] = {
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3,
        3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
    std::uint32_t i = 28;
    while (i > 0 && kBase[i] > len) --i;
    LenCode c;
    c.sym = static_cast<std::uint16_t>(257u + i);
    c.extra = kExtra[i];
    c.base = kBase[i];
    return c;
}

struct DistCode { std::uint16_t sym; std::uint8_t extra; std::uint16_t base; };

DistCode dist_code_for(std::uint32_t d) noexcept {
    assert(d >= 1u && d <= kWindow);
    static const std::uint16_t kBase[30] = {
        1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385,
        513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385,
        24577};
    static const std::uint8_t kExtra[30] = {
        0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7,
        8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
    std::uint32_t i = 29;
    while (i > 0 && kBase[i] > d) --i;
    DistCode c;
    c.sym = static_cast<std::uint16_t>(i);
    c.extra = kExtra[i];
    c.base = kBase[i];
    return c;
}

inline std::uint32_t hash3(const std::uint8_t* p) noexcept {
    const std::uint32_t v = (static_cast<std::uint32_t>(p[0]) << 16) |
                            (static_cast<std::uint32_t>(p[1]) << 8) |
                            static_cast<std::uint32_t>(p[2]);
    return (v * 2654435761u) >> (32u - kHashBits);
}

// A STORED block: BFINAL/BTYPE, byte-align, LEN, ~LEN, raw bytes.
bool emit_stored(BitW* w, const std::uint8_t* src, std::uint32_t n,
                 bool final_block) noexcept {
    bw_put(w, final_block ? 1u : 0u, 1u);
    bw_put(w, 0u, 2u);                       // BTYPE = 00
    bw_align(w);
    if (!w->ok) return false;
    if (w->pos + 4u + n > w->cap) { w->ok = false; return false; }
    w->dst[w->pos++] = static_cast<std::uint8_t>(n & 0xFFu);
    w->dst[w->pos++] = static_cast<std::uint8_t>((n >> 8) & 0xFFu);
    const std::uint16_t inv = static_cast<std::uint16_t>(~n);
    w->dst[w->pos++] = static_cast<std::uint8_t>(inv & 0xFFu);
    w->dst[w->pos++] = static_cast<std::uint8_t>((inv >> 8) & 0xFFu);
    if (n != 0u) std::memcpy(w->dst + w->pos, src, n);
    w->pos += n;
    return true;
}

}  // namespace

// ---- CRC-32 ---------------------------------------------------------------

std::uint32_t crc32_ieee(const std::uint8_t* data, std::uint64_t len,
                         std::uint32_t seed) noexcept {
    assert(data != nullptr || len == 0);
    // Table built once on first use. 1 KiB, and the alternative -- a bitwise
    // loop -- is 8x slower on the page-sized inputs this sees.
    static std::uint32_t table[256];
    static bool built = false;
    if (!built) {
        for (std::uint32_t i = 0; i < 256u; ++i) {
            std::uint32_t c = i;
            for (int k = 0; k < 8; ++k) {
                c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            }
            table[i] = c;
        }
        built = true;
    }
    std::uint32_t c = seed ^ 0xFFFFFFFFu;
    for (std::uint64_t i = 0; i < len; ++i) {
        c = table[(c ^ data[i]) & 0xFFu] ^ (c >> 8);
    }
    return c ^ 0xFFFFFFFFu;
}

// ---- DEFLATE --------------------------------------------------------------

bool deflate_raw_compress(const std::uint8_t* src, std::uint64_t src_len,
                          std::uint8_t* dst, std::uint64_t dst_cap,
                          std::uint64_t* out_len, DeflateState* st) noexcept {
    assert(out_len != nullptr);
    assert(src != nullptr || src_len == 0);
    if (dst == nullptr || st == nullptr) return false;
    if (src_len > 0xFFFFFFFFull) return false;

    BitW w{dst, dst_cap, 0, 0, 0, true};
    const std::uint32_t n = static_cast<std::uint32_t>(src_len);

    if (n == 0u) {
        // An empty stream still needs a final block, or an inflater waits
        // forever for one.
        bw_put(&w, 1u, 1u);
        bw_put(&w, 1u, 2u);                  // fixed Huffman
        bw_sym(&w, 256u);                    // end-of-block
        bw_align(&w);
        if (!w.ok) return false;
        *out_len = w.pos;
        return true;
    }

    // kNoPos, not 0: position 0 is a perfectly good match candidate, and
    // using it as the sentinel silently drops every match against the start
    // of the input. 0xFF fill sets every entry to kNoPos.
    std::memset(st->head, 0xFF, sizeof(st->head));

    // Emit one fixed-Huffman block covering the whole input. Splitting into
    // several would let each pick its own block type; the STORED fallback
    // below handles the case that actually matters (incompressible input)
    // by measuring the result and re-emitting.
    bw_put(&w, 1u, 1u);                      // BFINAL
    bw_put(&w, 1u, 2u);                      // BTYPE = 01, fixed Huffman

    std::uint32_t i = 0;
    while (i < n) {
        std::uint32_t best_len = 0, best_dist = 0;
        if (i + kMinMatch <= n) {
            const std::uint32_t h = hash3(src + i);
            std::uint32_t cand = st->head[h];
            std::uint32_t chain = 0;
            const std::uint32_t max_len =
                (n - i < kMaxMatch) ? (n - i) : kMaxMatch;
            while (cand != kNoPos && chain < kMaxChain) {
                const std::uint32_t d = i - cand;
                if (d == 0u || d > kWindow) break;
                // Compare only if it can beat what we have -- checking the
                // byte one past the current best first rejects most
                // candidates in one load.
                if (src[cand + best_len] == src[i + best_len]) {
                    std::uint32_t l = 0;
                    while (l < max_len && src[cand + l] == src[i + l]) ++l;
                    if (l > best_len) {
                        best_len = l;
                        best_dist = d;
                        if (l >= max_len) break;   // cannot do better
                    }
                }
                cand = st->prev[cand & (kWindow - 1u)];
                ++chain;
            }
            // Insert AFTER searching so a position never matches itself.
            st->prev[i & (kWindow - 1u)] = st->head[h];
            st->head[h] = i;
        }

        if (best_len >= kMinMatch) {
            const LenCode lc = len_code_for(best_len);
            bw_sym(&w, lc.sym);
            if (lc.extra != 0u) {
                bw_put(&w, best_len - lc.base, lc.extra);
            }
            const DistCode dc = dist_code_for(best_dist);
            bw_dist_sym(&w, dc.sym);
            if (dc.extra != 0u) {
                bw_put(&w, best_dist - dc.base, dc.extra);
            }
            // Index the positions the match covers so later matches can
            // reference inside it. Bounded by the match length.
            for (std::uint32_t k = 1; k < best_len; ++k) {
                const std::uint32_t p = i + k;
                if (p + kMinMatch > n) break;
                const std::uint32_t hh = hash3(src + p);
                st->prev[p & (kWindow - 1u)] = st->head[hh];
                st->head[hh] = p;
            }
            i += best_len;
        } else {
            bw_sym(&w, src[i]);
            ++i;
        }
        if (!w.ok) break;
    }
    bw_sym(&w, 256u);                        // end-of-block
    bw_align(&w);

    // If the fixed-Huffman block did not fit, or did not actually shrink the
    // input, fall back to STORED blocks. A naive fixed-Huffman-only encoder
    // EXPANDS incompressible data by roughly 12%; storing it costs 5 bytes
    // per 64 KiB instead.
    if (!w.ok || w.pos >= src_len) {
        BitW s{dst, dst_cap, 0, 0, 0, true};
        std::uint32_t off = 0;
        do {
            const std::uint32_t take =
                (n - off > kMaxStored) ? kMaxStored : (n - off);
            const bool last = (off + take >= n);
            if (!emit_stored(&s, src + off, take, last)) return false;
            off += take;
        } while (off < n);
        if (!s.ok) return false;
        *out_len = s.pos;
        return true;
    }
    *out_len = w.pos;
    return true;
}

bool gzip_compress(const std::uint8_t* src, std::uint64_t src_len,
                   std::uint8_t* dst, std::uint64_t dst_cap,
                   std::uint64_t* out_len, DeflateState* st) noexcept {
    assert(out_len != nullptr);
    if (dst == nullptr) return false;
    if (dst_cap < 18u) return false;
    // RFC 1952 header: magic, CM=8 (deflate), FLG=0, MTIME=0 (deliberately:
    // a timestamp would make output non-reproducible), XFL=0, OS=255
    // (unknown -- claiming a specific OS in a columnar page is noise).
    dst[0] = 0x1F; dst[1] = 0x8B; dst[2] = 8; dst[3] = 0;
    dst[4] = 0; dst[5] = 0; dst[6] = 0; dst[7] = 0;
    dst[8] = 0; dst[9] = 255;
    std::uint64_t body = 0;
    if (!deflate_raw_compress(src, src_len, dst + 10, dst_cap - 18u, &body,
                              st)) {
        return false;
    }
    const std::uint32_t crc = crc32_ieee(src, src_len, 0);
    const std::uint32_t isize = static_cast<std::uint32_t>(src_len);
    std::uint64_t p = 10u + body;
    dst[p++] = static_cast<std::uint8_t>(crc & 0xFFu);
    dst[p++] = static_cast<std::uint8_t>((crc >> 8) & 0xFFu);
    dst[p++] = static_cast<std::uint8_t>((crc >> 16) & 0xFFu);
    dst[p++] = static_cast<std::uint8_t>((crc >> 24) & 0xFFu);
    dst[p++] = static_cast<std::uint8_t>(isize & 0xFFu);
    dst[p++] = static_cast<std::uint8_t>((isize >> 8) & 0xFFu);
    dst[p++] = static_cast<std::uint8_t>((isize >> 16) & 0xFFu);
    dst[p++] = static_cast<std::uint8_t>((isize >> 24) & 0xFFu);
    *out_len = p;
    return true;
}

}  // namespace ingest
}  // namespace bolt
