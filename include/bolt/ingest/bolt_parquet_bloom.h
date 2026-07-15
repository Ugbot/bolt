// bolt/ingest/bolt_parquet_bloom.h — Parquet split-block bloom filter
// (SBBF) reader + membership test (G2FEAT-23, bolt half).
//
// Per parquet-format BloomFilter.md, a column chunk's bloom filter lives
// at ColumnMetaData::bloom_filter_offset (captured on PqChunk by
// bolt_parquet_meta.h) as a thrift-compact BloomFilterHeader followed
// immediately by the raw bitset bytes:
//
//   BloomFilterHeader { 1: i32 numBytes,
//                       2: union algorithm   { 1: BLOCK }
//                       3: union hash        { 1: XXHASH }
//                       4: union compression { 1: UNCOMPRESSED } }
//
// Only the spec's one defined combination (BLOCK + XXHASH + UNCOMPRESSED)
// is accepted; any other union arm fails the parse (conservative — the
// caller simply doesn't get a bloom filter, never a wrong answer).
//
// The bitset is an array of 256-bit blocks (8 x u32 little-endian words).
// A value's 64-bit key is XXH64(plain-encoded value, seed = 0):
//   INT32/INT64  -> 4/8-byte little-endian two's complement
//   BYTE_ARRAY   -> the raw bytes, NO length prefix
// Membership: block = fastrange(top 32 bits of key, numBytes/32); inside
// the block each of the 8 words must have bit ((lo32 * salt[w]) >> 27)
// set. False positives are possible by design; false negatives are not —
// pq_bloom_may_contain() == false PROVES the value is absent from the
// chunk, so a point predicate (=, IN element) may skip the whole chunk.
//
// Tiger Style: POD outputs, noexcept, no allocation (bitset stays a view
// into the caller's file buffer), bounded loops, hostile input -> false.

#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>

#include "bolt/ingest/bolt_parquet_meta.h"

namespace bolt {
namespace ingest {
namespace parquet {

// Hostile-input cap on the bitset size. parquet-mr's default writer cap
// is 1 MiB; 64 MiB admits any sane writer while bounding a corrupt
// numBytes against the file slice.
inline constexpr uint32_t kPqBloomMaxBytes = 64u << 20;

// A parsed, validated split-block bloom filter. `bitset` is a VIEW into
// the caller's file buffer (no copy) — it must outlive the filter.
struct PqBloomFilter {
    const uint8_t* bitset;
    uint32_t       n_bytes;       // multiple of 32, >= 32
    uint32_t       _pad;
};

// Parse + validate a BloomFilterHeader at buf[0..len) and locate the
// bitset that follows it. On success *consumed = header byte count and
// out->bitset/n_bytes are set (bitset fully inside the slice). False on
// any unsupported/corrupt shape — never UB on hostile input.
bool pq_parse_bloom(const uint8_t* buf, uint64_t len,
                    PqBloomFilter* out, uint64_t* consumed) noexcept;

// Convenience: locate the chunk's bloom region in the whole-file buffer
// and parse. Uses bloom_filter_length when present; pre-2.9 writers omit
// it, in which case the slice is bounded by end-of-file (the header's
// own numBytes then bounds the bitset).
bool pq_read_bloom(const uint8_t* file, uint64_t file_len,
                   const PqChunk& ch, PqBloomFilter* out) noexcept;

// XXH64 (xxhash.com, the spec-mandated bloom hash). Public because the
// caller hashes predicate literals once and probes many chunks.
uint64_t pq_xxh64(const uint8_t* data, uint64_t len, uint64_t seed) noexcept;

// Spec plain-encoding hash helpers (seed 0, per parquet-format).
inline uint64_t pq_bloom_hash_i64(int64_t v) noexcept {
    uint8_t b[8];
    std::memcpy(b, &v, 8);                 // two's complement little-endian
    return pq_xxh64(b, 8, 0);
}

inline uint64_t pq_bloom_hash_i32(int32_t v) noexcept {
    uint8_t b[4];
    std::memcpy(b, &v, 4);
    return pq_xxh64(b, 4, 0);
}

inline uint64_t pq_bloom_hash_bytes(const uint8_t* p, uint32_t n) noexcept {
    assert(p != nullptr || n == 0);
    return pq_xxh64(p, n, 0);
}

// Split-block membership probe. False = PROVEN absent (safe to skip);
// true = may be present (must read).
bool pq_bloom_may_contain(const PqBloomFilter& bf, uint64_t key) noexcept;

}  // namespace parquet
}  // namespace ingest
}  // namespace bolt
