// bolt/ingest/bolt_roaring.h — Roaring bitmap deserialiser (portable format).
//
// Delta deletion vectors and Iceberg position-delete bitmaps ship as
// CRoaring "portable" serialised bitmaps. This reads that format into an
// arena-allocated POD `RoaringBitmap` and answers membership / cardinality.
// Deserialise-only — we never write Roaring bitmaps here.
//
// Portable 32-bit format (CRoaring portable spec):
//   [u32 cookie]   either SERIAL_COOKIE_NO_RUNCONTAINER (0x00003B4D, no runs)
//                  or (SERIAL_COOKIE=0x0000373B | ((n_containers-1)<<16)) when
//                  a run-flag bitset is present.
//   if cookie has run flag: [ceil(n/8) bytes] run-flag bitset, bit i set ⇒
//                  container i is a Run container.
//   [u32 size]     when cookie == NO_RUNCONTAINER (n_containers).
//   keyscards[n]:  n × { u16 key (high 16 bits), u16 cardinality-1 }.
//   if NO_RUNCONTAINER and n >= NO_OFFSET_THRESHOLD(4): [u32 offset]×n header.
//   payloads in container order:
//       Array  : (card) × u16 sorted values.
//       Bitset : 1024 × u64 words (8192 bytes).
//       Run    : [u16 n_runs] then n_runs × { u16 start, u16 length-1 }.
//
// 64-bit portable format (CRoaring "frozen"/portable r64):
//   [u64 n_buckets] then n_buckets × { u32 high32 key, <32-bit portable map> }.
//   `roaring_deserialize_r64` parses that; the 32-bit reader handles one map.
//
// Tiger Style: PODs, ≥2 asserts/fn, bounded loops, no exceptions, Arena allocs.

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "bolt/bolt_arena.h"

namespace bolt {
namespace ingest {

// Cookies + thresholds from the CRoaring portable spec.
static constexpr uint32_t kRoaringCookieNoRun  = 0x00003B4Du;   // 12345 + ...
static constexpr uint32_t kRoaringCookieRun     = 0x0000373Bu;
static constexpr uint32_t kRoaringNoOffsetThr   = 4u;
// 32-bit portable cookie callers may probe (low 16 bits when run-flag set).
static constexpr uint32_t kRoaringSerialCookie  = 0x0000373Bu;

static constexpr uint32_t kRoaringMaxContainers = 1u << 16;     // key is u16

enum class RoaringKind : uint8_t { kArray = 0, kBitset = 1, kRun = 2 };

// One container's decoded payload. Pointers are into the arena; the bitset
// path keeps the raw 1024-word slab, the array/run paths keep u16 slices.
struct RoaringContainer {
    uint16_t     key;            // high 16 bits of every value in this block
    RoaringKind  kind;
    uint8_t      _pad;
    uint32_t     cardinality;    // exact popcount of the container
    const uint16_t* values;      // Array: card entries.  Run: 2*n_runs entries
                                 // laid out as (start,length-1) pairs.
    const uint64_t* words;       // Bitset: 1024 words.  nullptr otherwise.
    uint32_t     n_runs;         // Run: number of runs.  0 otherwise.
    uint32_t     _pad2;
};

// A deserialised 32-bit bitmap. POD; arena owns the container array + payloads.
struct RoaringBitmap {
    RoaringContainer* containers;
    uint32_t          n_containers;
    uint32_t          _pad;
    uint64_t          cardinality;     // sum of container cardinalities
};

// Parse a portable 32-bit serialised bitmap from src[0..src_len). All storage
// is bump-allocated into `arena`. Returns false on any malformed shape (bad
// cookie, truncation, container overflow) — never UB.
bool roaring_deserialize(const uint8_t* src, uint64_t src_len,
                         Arena* arena, RoaringBitmap* out) noexcept;

// Membership test: is the 32-bit value present?
bool roaring_contains(const RoaringBitmap* bm, uint32_t value) noexcept;

// Total set-bit count.
uint64_t roaring_cardinality(const RoaringBitmap* bm) noexcept;

// ---------------------------------------------------------------------------
// 64-bit portable variant — n_buckets × (high32, 32-bit map). Each sub-map is
// decoded into its own RoaringBitmap; the buckets array is arena-allocated.
// ---------------------------------------------------------------------------
struct RoaringBucket {
    uint32_t      high32;
    uint32_t      _pad;
    RoaringBitmap map;
};

struct RoaringBitmap64 {
    RoaringBucket* buckets;
    uint32_t       n_buckets;
    uint32_t       _pad;
    uint64_t       cardinality;
};

bool roaring_deserialize_r64(const uint8_t* src, uint64_t src_len,
                             Arena* arena, RoaringBitmap64* out) noexcept;
bool roaring_contains64(const RoaringBitmap64* bm, uint64_t value) noexcept;
uint64_t roaring_cardinality64(const RoaringBitmap64* bm) noexcept;

}  // namespace ingest
}  // namespace bolt
