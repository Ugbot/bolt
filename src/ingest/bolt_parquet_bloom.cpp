// bolt_parquet_bloom.cpp — Parquet split-block bloom filter (SBBF) reader
// + XXH64. See bolt_parquet_bloom.h for scope (G2FEAT-23, bolt half).

#include "bolt/ingest/bolt_parquet_bloom.h"

namespace bolt {
namespace ingest {
namespace parquet {

// ---- XXH64 ------------------------------------------------------------------
// Straight from the published xxHash specification (xxhash.com); verified
// against the canonical test vectors in test_bolt_parquet_pageindex.cpp
// and end-to-end against pyarrow- and duckdb-written bloom filters.

namespace {

constexpr uint64_t kXxP1 = 0x9E3779B185EBCA87ull;
constexpr uint64_t kXxP2 = 0xC2B2AE3D27D4EB4Full;
constexpr uint64_t kXxP3 = 0x165667B19E3779F9ull;
constexpr uint64_t kXxP4 = 0x85EBCA77C2B2AE63ull;
constexpr uint64_t kXxP5 = 0x27D4EB2F165667C5ull;

inline uint64_t xx_rotl(uint64_t v, uint32_t n) noexcept {
    return (v << n) | (v >> (64u - n));
}

inline uint64_t xx_read64(const uint8_t* p) noexcept {
    uint64_t v;
    std::memcpy(&v, p, 8);                  // little-endian hosts only (MSVC/
    return v;                               // clang/gcc x64+arm64 — bolt's
}                                           // supported matrix)

inline uint32_t xx_read32(const uint8_t* p) noexcept {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

inline uint64_t xx_round(uint64_t acc, uint64_t input) noexcept {
    return xx_rotl(acc + input * kXxP2, 31) * kXxP1;
}

inline uint64_t xx_merge_round(uint64_t acc, uint64_t val) noexcept {
    acc ^= xx_round(0, val);
    return acc * kXxP1 + kXxP4;
}

}  // namespace

uint64_t pq_xxh64(const uint8_t* data, uint64_t len, uint64_t seed) noexcept {
    assert(data != nullptr || len == 0);
    const uint8_t* p = data;
    const uint8_t* const end = data + len;
    uint64_t h;
    if (len >= 32) {
        uint64_t v1 = seed + kXxP1 + kXxP2;
        uint64_t v2 = seed + kXxP2;
        uint64_t v3 = seed;
        uint64_t v4 = seed - kXxP1;
        do {                                 // bounded: len / 32 stripes
            v1 = xx_round(v1, xx_read64(p));      p += 8;
            v2 = xx_round(v2, xx_read64(p));      p += 8;
            v3 = xx_round(v3, xx_read64(p));      p += 8;
            v4 = xx_round(v4, xx_read64(p));      p += 8;
        } while (p + 32 <= end);
        h = xx_rotl(v1, 1) + xx_rotl(v2, 7) + xx_rotl(v3, 12) +
            xx_rotl(v4, 18);
        h = xx_merge_round(h, v1);
        h = xx_merge_round(h, v2);
        h = xx_merge_round(h, v3);
        h = xx_merge_round(h, v4);
    } else {
        h = seed + kXxP5;
    }
    h += len;
    while (p + 8 <= end) {
        h ^= xx_round(0, xx_read64(p));
        h = xx_rotl(h, 27) * kXxP1 + kXxP4;
        p += 8;
    }
    if (p + 4 <= end) {
        h ^= static_cast<uint64_t>(xx_read32(p)) * kXxP1;
        h = xx_rotl(h, 23) * kXxP2 + kXxP3;
        p += 4;
    }
    while (p < end) {
        h ^= static_cast<uint64_t>(*p) * kXxP5;
        h = xx_rotl(h, 11) * kXxP1;
        ++p;
    }
    h ^= h >> 33;
    h *= kXxP2;
    h ^= h >> 29;
    h *= kXxP3;
    h ^= h >> 32;
    return h;
}

// ---- BloomFilterHeader ------------------------------------------------------

namespace {

// Parse one of the header's three thrift unions. Accepts ONLY the spec's
// single defined arm (field id 1, an empty struct); any other arm or a
// multi-arm union fails.
bool parse_union_arm1(TcCursor* c) noexcept {
    assert(c != nullptr);
    int16_t fid = 0;
    uint8_t ft;
    bool saw_arm1 = false;
    while (tc_field(c, &fid, &ft)) {
        if (fid == 1 && ft == kTcStruct && !saw_arm1) {
            // The arm is an empty struct: consume fields until STOP.
            int16_t ifid = 0;
            uint8_t ift;
            while (tc_field(c, &ifid, &ift)) {
                if (!tc_skip(c, ift, 0)) return false;
            }
            saw_arm1 = true;
        } else {
            return false;                    // unsupported arm / duplicate
        }
    }
    return saw_arm1;
}

}  // namespace

bool pq_parse_bloom(const uint8_t* buf, uint64_t len,
                    PqBloomFilter* out, uint64_t* consumed) noexcept {
    assert(out != nullptr && consumed != nullptr);
    if (buf == nullptr || len == 0) return false;
    out->bitset = nullptr;
    out->n_bytes = 0;
    TcCursor c{buf, buf + len};
    int16_t fid = 0;
    uint8_t ft;
    int64_t num_bytes = -1;
    bool algo_ok = false, hash_ok = false, comp_ok = false;
    while (tc_field(&c, &fid, &ft)) {
        switch (fid) {
            case 1: {   // numBytes: i32
                if (!tc_zigzag(&c, &num_bytes)) return false;
                break;
            }
            case 2:     // algorithm union: BLOCK
                if (ft != kTcStruct || !parse_union_arm1(&c)) return false;
                algo_ok = true;
                break;
            case 3:     // hash union: XXHASH
                if (ft != kTcStruct || !parse_union_arm1(&c)) return false;
                hash_ok = true;
                break;
            case 4:     // compression union: UNCOMPRESSED
                if (ft != kTcStruct || !parse_union_arm1(&c)) return false;
                comp_ok = true;
                break;
            default:
                if (!tc_skip(&c, ft, 0)) return false;
                break;
        }
    }
    if (!algo_ok || !hash_ok || !comp_ok) return false;
    if (num_bytes < 32 ||
        num_bytes > static_cast<int64_t>(kPqBloomMaxBytes)) return false;
    if ((num_bytes % 32) != 0) return false;             // whole 256-bit blocks
    const uint64_t header_bytes = static_cast<uint64_t>(c.p - buf);
    if (static_cast<uint64_t>(num_bytes) > len - header_bytes) return false;
    out->bitset = buf + header_bytes;
    out->n_bytes = static_cast<uint32_t>(num_bytes);
    *consumed = header_bytes;
    return true;
}

bool pq_read_bloom(const uint8_t* file, uint64_t file_len,
                   const PqChunk& ch, PqBloomFilter* out) noexcept {
    assert(out != nullptr);
    if (file == nullptr || ch.bloom_filter_offset <= 0) return false;
    const uint64_t off = static_cast<uint64_t>(ch.bloom_filter_offset);
    if (off >= file_len) return false;
    uint64_t avail = file_len - off;
    if (ch.bloom_filter_length > 0) {                    // modern writers
        const uint64_t bl = static_cast<uint64_t>(ch.bloom_filter_length);
        if (bl < avail) avail = bl;
    }
    uint64_t consumed = 0;
    return pq_parse_bloom(file + off, avail, out, &consumed);
}

// ---- membership -------------------------------------------------------------

bool pq_bloom_may_contain(const PqBloomFilter& bf, uint64_t key) noexcept {
    assert(bf.bitset != nullptr);
    assert(bf.n_bytes >= 32u && (bf.n_bytes % 32u) == 0u);
    // parquet-format BloomFilter.md salt constants (block_check).
    static constexpr uint32_t kSalt[8] = {
        0x47b6137bU, 0x44974d91U, 0x8824ad5bU, 0xa2b7289dU,
        0x705495c7U, 0x2df1424bU, 0x9efc4947U, 0x5c6bfb31U,
    };
    const uint32_t n_blocks = bf.n_bytes / 32u;
    // fastrange: map the TOP 32 bits of the key onto [0, n_blocks).
    const uint64_t block_idx =
        (static_cast<uint64_t>(static_cast<uint32_t>(key >> 32)) *
         static_cast<uint64_t>(n_blocks)) >> 32;
    const uint8_t* block = bf.bitset + block_idx * 32u;
    const uint32_t x = static_cast<uint32_t>(key);       // low 32 bits
    for (uint32_t w = 0; w < 8u; ++w) {                  // bounded: 8 words
        uint32_t word;
        std::memcpy(&word, block + 4u * w, 4);           // little-endian
        const uint32_t bit = (x * kSalt[w]) >> 27;       // 0..31
        if ((word & (1u << bit)) == 0u) return false;    // proven absent
    }
    return true;
}

}  // namespace parquet
}  // namespace ingest
}  // namespace bolt
