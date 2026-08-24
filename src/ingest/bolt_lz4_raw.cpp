// bolt_lz4_raw.cpp — self-contained LZ4 block codec. See the header for the
// format and for why bolt_lz4.h's liblz4 shim is not enough.

#include "bolt/ingest/bolt_lz4_raw.h"

#include <cassert>
#include <cstring>

namespace bolt {
namespace ingest {

namespace {

// A match is at least 4 bytes (the token's low nibble encodes length - 4).
constexpr uint32_t kMinMatch = 4u;
// The last match must begin at least this far from the end of the block, and
// the final 5 bytes are always literals. Both are spec rules, not tuning:
// a decoder is allowed to copy in wide chunks near the end only because of
// them, and liblz4 rejects a block that breaks either.
constexpr uint32_t kLastLiterals = 5u;
constexpr uint32_t kMfLimit = 12u;
// 16-bit offsets: a match can reach at most this far back.
constexpr uint32_t kMaxDistance = 65535u;

// Hash table over 4-byte sequences. 4096 entries (16 KiB) keeps the table in
// L1; liblz4's default is 16 KiB for the same reason. A bigger table finds
// more matches and costs cache misses on every insert, which is the wrong
// trade for page-sized inputs.
constexpr uint32_t kHashLog = 12u;
constexpr uint32_t kHashSize = 1u << kHashLog;

inline uint32_t read32(const uint8_t* p) noexcept {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

inline uint32_t hash4(uint32_t v) noexcept {
    // Knuth multiplicative, folded to kHashLog bits. Cheap and adequate for a
    // greedy matcher -- a collision costs one failed 4-byte compare.
    return (v * 2654435761u) >> (32u - kHashLog);
}

// Emit an extended length: 0xFF bytes then the remainder. Returns false if it
// would not fit.
inline bool emit_len(uint8_t* dst, uint64_t cap, uint64_t* op,
                     uint32_t len) noexcept {
    while (len >= 255u) {
        if (*op >= cap) return false;
        dst[(*op)++] = 255u;
        len -= 255u;
    }
    if (*op >= cap) return false;
    dst[(*op)++] = static_cast<uint8_t>(len);
    return true;
}

// Read an extended length. Bounded by the input; returns false on overrun or
// on a length that would overflow 32 bits.
inline bool read_len(const uint8_t* src, uint64_t src_len, uint64_t* ip,
                     uint32_t* len) noexcept {
    uint32_t acc = *len;
    for (;;) {                                  // bounded by src_len
        if (*ip >= src_len) return false;
        const uint8_t b = src[(*ip)++];
        if (acc > 0xFFFFFFFFu - b) return false;   // length overflow
        acc += b;
        if (b != 255u) break;
    }
    *len = acc;
    return true;
}

}  // namespace

bool lz4_raw_compress(const uint8_t* src, uint64_t src_len,
                      uint8_t* dst, uint64_t dst_cap,
                      uint64_t* out_len) noexcept {
    assert(out_len != nullptr);
    assert(src != nullptr || src_len == 0);
    if (dst == nullptr) return false;
    if (src_len > (uint64_t{1} << 31)) return false;

    uint64_t op = 0;
    // A block too short to hold a match is all literals by definition.
    if (src_len < kMfLimit + kMinMatch) {
        const uint32_t ll = static_cast<uint32_t>(src_len);
        if (op >= dst_cap) return false;
        dst[op++] = static_cast<uint8_t>((ll < 15u ? ll : 15u) << 4);
        if (ll >= 15u && !emit_len(dst, dst_cap, &op, ll - 15u)) return false;
        if (op + ll > dst_cap) return false;
        if (ll != 0u) std::memcpy(dst + op, src, ll);
        op += ll;
        *out_len = op;
        return true;
    }

    // Zeroed table means "no candidate"; position 0 is therefore never a
    // candidate, which costs at most one missed match at the very start.
    uint32_t table[kHashSize];
    std::memset(table, 0, sizeof(table));

    const uint32_t n = static_cast<uint32_t>(src_len);
    const uint32_t mflimit = n - kMfLimit;
    uint32_t ip = 0;          // scan cursor
    uint32_t anchor = 0;      // start of the pending literal run

    while (ip < mflimit) {
        // Greedy search with step acceleration: after a stretch of misses,
        // sample less densely. Same shape as liblz4's fast scan -- on
        // incompressible data this is what keeps the compressor from being
        // O(n) hash inserts per byte.
        uint32_t match = 0;
        bool found = false;
        uint32_t step = 1;
        uint32_t searches = 0;
        while (ip < mflimit) {
            const uint32_t h = hash4(read32(src + ip));
            const uint32_t cand = table[h];
            table[h] = ip;
            if (cand != 0u && ip - cand <= kMaxDistance &&
                read32(src + cand) == read32(src + ip)) {
                match = cand;
                found = true;
                break;
            }
            ++searches;
            step = 1u + (searches >> 6);        // widen the stride slowly
            ip += step;
        }
        if (!found) break;

        // Extend the match backwards over literals already pending: those
        // bytes are cheaper as part of the match than as literals.
        while (ip > anchor && match > 0 && src[ip - 1] == src[match - 1]) {
            --ip;
            --match;
        }
        // Extend forwards, stopping short of the mandatory trailing literals.
        uint32_t mlen = kMinMatch;
        const uint32_t limit = n - kLastLiterals;
        while (ip + mlen < limit && src[match + mlen] == src[ip + mlen]) ++mlen;

        const uint32_t lit = ip - anchor;
        const uint32_t mcode = mlen - kMinMatch;

        // token
        if (op >= dst_cap) return false;
        dst[op++] = static_cast<uint8_t>(((lit < 15u ? lit : 15u) << 4) |
                                         (mcode < 15u ? mcode : 15u));
        if (lit >= 15u && !emit_len(dst, dst_cap, &op, lit - 15u)) return false;
        if (op + lit > dst_cap) return false;
        if (lit != 0u) std::memcpy(dst + op, src + anchor, lit);
        op += lit;
        // offset, little-endian
        const uint32_t off = ip - match;
        assert(off >= 1u && off <= kMaxDistance);
        if (op + 2u > dst_cap) return false;
        dst[op++] = static_cast<uint8_t>(off & 0xFFu);
        dst[op++] = static_cast<uint8_t>((off >> 8) & 0xFFu);
        if (mcode >= 15u && !emit_len(dst, dst_cap, &op, mcode - 15u)) {
            return false;
        }

        ip += mlen;
        anchor = ip;
        // Index the two positions just consumed so a following match can
        // reference inside this one.
        if (ip < mflimit) {
            table[hash4(read32(src + ip - 2u))] = ip - 2u;
        }
    }

    // Trailing literals: everything from the last match to the end.
    const uint32_t lit = n - anchor;
    if (op >= dst_cap) return false;
    dst[op++] = static_cast<uint8_t>((lit < 15u ? lit : 15u) << 4);
    if (lit >= 15u && !emit_len(dst, dst_cap, &op, lit - 15u)) return false;
    if (op + lit > dst_cap) return false;
    if (lit != 0u) std::memcpy(dst + op, src + anchor, lit);
    op += lit;
    *out_len = op;
    return true;
}

bool lz4_raw_decompress(const uint8_t* src, uint64_t src_len,
                        uint8_t* dst, uint64_t dst_cap) noexcept {
    assert(src != nullptr || src_len == 0);
    if (dst == nullptr && dst_cap != 0) return false;
    uint64_t ip = 0, op = 0;
    while (ip < src_len) {                       // bounded: ip strictly grows
        const uint8_t token = src[ip++];
        uint32_t ll = static_cast<uint32_t>(token >> 4);
        if (ll == 15u && !read_len(src, src_len, &ip, &ll)) return false;
        // literals
        if (ll != 0u) {
            if (ll > src_len - ip) return false;
            if (ll > dst_cap - op) return false;
            std::memcpy(dst + op, src + ip, ll);
            ip += ll;
            op += ll;
        }
        if (ip == src_len) break;                // last sequence: literals only
        // offset
        if (src_len - ip < 2u) return false;
        const uint32_t off = static_cast<uint32_t>(src[ip]) |
                             (static_cast<uint32_t>(src[ip + 1u]) << 8);
        ip += 2u;
        if (off == 0u || off > op) return false;  // reaches before the output
        uint32_t ml = static_cast<uint32_t>(token & 0x0Fu);
        if (ml == 15u && !read_len(src, src_len, &ip, &ml)) return false;
        if (ml > 0xFFFFFFFFu - kMinMatch) return false;
        ml += kMinMatch;
        if (ml > dst_cap - op) return false;
        // Byte-at-a-time because a match MAY overlap the cursor (off < ml is
        // how LZ4 encodes a run); memcpy/memmove would both be wrong.
        const uint8_t* m = dst + op - off;
        uint8_t* o = dst + op;
        for (uint32_t i = 0; i < ml; ++i) o[i] = m[i];
        op += ml;
    }
    // Parquet knows the exact uncompressed size; a block that yields anything
    // else is corrupt, and accepting it would hand the caller a partly
    // uninitialised buffer.
    return op == dst_cap;
}

}  // namespace ingest
}  // namespace bolt
