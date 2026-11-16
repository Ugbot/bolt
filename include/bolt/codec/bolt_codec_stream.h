// bolt_codec_stream.h — streams, byte buffers and the integer stages of the
// column codec chain (PROTOTYPE, see bolt_column_codec.h).
//
// A stream is one homogeneous run of a column chunk: opaque bytes, int64
// values, or strings (uint32 offsets[n + 1] + bytes). A stage turns one stream
// into one or more child streams; the chain recurses on each child.

#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>

#include "bolt/bolt_arena.h"
#include "bolt/kernels/bolt_fastlanes.h"

namespace bolt {
namespace codec {

inline constexpr uint64_t kMaxRows = 1u << 26;        // per encoded chunk
inline constexpr uint64_t kMaxChunkBytes = 1ull << 31;

enum class Kind : uint8_t { kBytes = 0, kInts = 1, kStrs = 2, kStrsSorted = 3 };

struct Stream {
    Kind kind = Kind::kBytes;
    uint64_t n = 0;                  // rows (kBytes: byte count)
    const uint8_t* bytes = nullptr;  // kBytes / string payload
    uint64_t nbytes = 0;
    const int64_t* ints = nullptr;   // kInts
    const uint32_t* offs = nullptr;  // strings: n + 1 offsets into bytes
};

inline Stream bytes_stream(const uint8_t* p, uint64_t n) noexcept {
    Stream s; s.kind = Kind::kBytes; s.n = n; s.bytes = p; s.nbytes = n; return s;
}
inline Stream ints_stream(const int64_t* v, uint64_t n) noexcept {
    Stream s; s.kind = Kind::kInts; s.n = n; s.ints = v; return s;
}
inline Stream strs_stream(const uint32_t* offs, const uint8_t* bytes, uint64_t n,
                          Kind k = Kind::kStrs) noexcept {
    assert(k == Kind::kStrs || k == Kind::kStrsSorted);
    Stream s; s.kind = k; s.n = n; s.offs = offs; s.bytes = bytes;
    s.nbytes = n ? offs[n] : 0; return s;
}

// Arena-backed growable byte buffer. A failed allocation latches ok = false.
struct Buf {
    uint8_t* p = nullptr;
    uint64_t n = 0, cap = 0;
    Arena* a = nullptr;
    bool ok = true;
};

inline bool buf_reserve(Buf& b, uint64_t extra) noexcept {
    assert(b.a != nullptr);
    if (b.n + extra <= b.cap) return b.ok;
    uint64_t cap = b.cap * 2 > b.n + extra ? b.cap * 2 : b.n + extra;
    if (cap < 256) cap = 256;
    auto* q = static_cast<uint8_t*>(b.a->allocate(cap, 8));
    if (q == nullptr) { b.ok = false; return false; }
    if (b.n) std::memcpy(q, b.p, b.n);
    b.p = q; b.cap = cap;
    return b.ok;
}
inline void buf_put(Buf& b, const void* src, uint64_t len) noexcept {
    if (len == 0 || !buf_reserve(b, len)) return;
    std::memcpy(b.p + b.n, src, len); b.n += len;
}
inline void buf_u8(Buf& b, uint8_t v) noexcept {
    if (!buf_reserve(b, 1)) return;
    b.p[b.n++] = v;
}
inline void buf_var(Buf& b, uint64_t v) noexcept {
    if (!buf_reserve(b, 10)) return;
    while (v >= 0x80) { b.p[b.n++] = static_cast<uint8_t>(v) | 0x80; v >>= 7; }
    b.p[b.n++] = static_cast<uint8_t>(v);
}

// Bounds-checked reader. Any short read latches ok = false and yields zeros.
struct Rd {
    const uint8_t* p = nullptr;
    const uint8_t* end = nullptr;
    bool ok = true;
};
inline uint8_t rd_u8(Rd& r) noexcept {
    if (r.p >= r.end) { r.ok = false; return 0; }
    return *r.p++;
}
inline uint64_t rd_var(Rd& r) noexcept {
    uint64_t v = 0;
    for (uint32_t sh = 0; sh < 64; sh += 7) {
        const uint8_t c = rd_u8(r);
        v |= static_cast<uint64_t>(c & 0x7f) << sh;
        if (!(c & 0x80)) return v;
    }
    r.ok = false; return 0;
}
inline const uint8_t* rd_bytes(Rd& r, uint64_t len) noexcept {
    if (static_cast<uint64_t>(r.end - r.p) < len) { r.ok = false; return nullptr; }
    const uint8_t* q = r.p; r.p += len; return q;
}

// ---- integer stages -------------------------------------------------------

inline void delta_encode(const int64_t* v, uint64_t n, int64_t* out) noexcept {
    assert(v != nullptr || n == 0);
    uint64_t prev = 0;
    for (uint64_t i = 0; i < n; ++i) {
        const uint64_t cur = static_cast<uint64_t>(v[i]);
        out[i] = static_cast<int64_t>(cur - prev); prev = cur;
    }
}
inline void delta_decode(const int64_t* d, uint64_t n, int64_t* out) noexcept {
    assert(d != nullptr || n == 0);
    uint64_t acc = 0;
    for (uint64_t i = 0; i < n; ++i) { acc += static_cast<uint64_t>(d[i]); out[i] = static_cast<int64_t>(acc); }
}

inline uint64_t tail_bytes(uint64_t m, uint32_t w) noexcept { return (m * w + 7) / 8; }

// Frame of reference in 1,024-value FastLanes blocks, each with its own
// (ref, width); a short last block is packed sequentially (no padding).
inline void for_encode(const int64_t* v, uint64_t n, Buf& out) noexcept {
    assert(v != nullptr || n == 0);
    alignas(64) uint64_t packed[fastlanes::kBlockValues];
    for (uint64_t at = 0; at < n; at += fastlanes::kBlockValues) {
        const uint64_t m = n - at < fastlanes::kBlockValues ? n - at : fastlanes::kBlockValues;
        int64_t ref = 0; uint32_t w = 0;
        fastlanes::choose_for<int64_t>(v + at, static_cast<int64_t>(m), &ref, &w);
        buf_put(out, &ref, 8); buf_u8(out, static_cast<uint8_t>(w));
        if (m == fastlanes::kBlockValues) {
            fastlanes::encode_for<int64_t>(v + at, static_cast<int64_t>(m), ref, w, packed);
            buf_put(out, packed, 128ull * w);
            continue;
        }
        const uint64_t nb = tail_bytes(m, w);
        if (!buf_reserve(out, nb)) return;
        std::memset(out.p + out.n, 0, nb);
        uint64_t pos = 0;
        for (uint64_t i = 0; i < m; ++i) {
            uint64_t x = static_cast<uint64_t>(v[at + i]) - static_cast<uint64_t>(ref);
            for (uint32_t left = w; left;) {
                const uint32_t off = static_cast<uint32_t>(pos & 7);
                const uint32_t take = 8 - off < left ? 8 - off : left;
                out.p[out.n + (pos >> 3)] |= static_cast<uint8_t>((x & ((1u << take) - 1u)) << off);
                x >>= take; pos += take; left -= take;
            }
        }
        out.n += nb;
    }
}

inline bool for_decode(Rd& r, uint64_t n, int64_t* out) noexcept {
    assert(out != nullptr || n == 0);
    alignas(64) uint64_t packed[fastlanes::kBlockValues];
    for (uint64_t at = 0; at < n; at += fastlanes::kBlockValues) {
        const uint64_t m = n - at < fastlanes::kBlockValues ? n - at : fastlanes::kBlockValues;
        const uint8_t* rp = rd_bytes(r, 8);
        const uint32_t w = rd_u8(r);
        if (!r.ok || w > 64) return false;
        int64_t ref; std::memcpy(&ref, rp, 8);
        if (m == fastlanes::kBlockValues) {
            const uint8_t* src = rd_bytes(r, 128ull * w);
            if (src == nullptr) return false;
            std::memcpy(packed, src, 128ull * w);
            fastlanes::decode_for<int64_t>(packed, static_cast<int64_t>(m), ref, w, out + at);
            continue;
        }
        const uint8_t* src = rd_bytes(r, tail_bytes(m, w));
        if (src == nullptr) return false;
        uint64_t pos = 0;
        for (uint64_t i = 0; i < m; ++i) {
            uint64_t x = 0;
            for (uint32_t got = 0; got < w;) {
                const uint32_t off = static_cast<uint32_t>(pos & 7);
                const uint32_t take = 8 - off < w - got ? 8 - off : w - got;
                x |= static_cast<uint64_t>((src[pos >> 3] >> off) & ((1u << take) - 1u)) << got;
                got += take; pos += take;
            }
            out[at + i] = static_cast<int64_t>(x + static_cast<uint64_t>(ref));
        }
    }
    return true;
}

// Run lengths. vals / lens hold at most n entries; returns the run count.
inline uint64_t rle_encode(const int64_t* v, uint64_t n, int64_t* vals, int64_t* lens) noexcept {
    assert(v != nullptr || n == 0);
    uint64_t runs = 0;
    for (uint64_t i = 0; i < n; ++i) {
        if (runs && vals[runs - 1] == v[i]) { ++lens[runs - 1]; continue; }
        vals[runs] = v[i]; lens[runs] = 1; ++runs;
    }
    return runs;
}

}  // namespace codec
}  // namespace bolt
