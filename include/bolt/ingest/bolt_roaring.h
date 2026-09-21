// bolt/ingest/bolt_roaring.h — Roaring bitmap deserialiser + serialiser
// (portable format).
//
// Delta deletion vectors and Iceberg position-delete bitmaps ship as
// CRoaring "portable" serialised bitmaps. This reads that format into an
// arena-allocated POD `RoaringBitmap` and answers membership / cardinality,
// and (G2ICE-80) writes it too — `roaring_serialize`/`roaring_serialize_r64`
// build a byte-exact portable bitmap from a sorted, deduplicated value list,
// verified against real CRoaring (`pyroaring`) and real delta-rs/DuckDB
// output, not only against this reader.
//
// Portable 32-bit format (CRoaring portable spec — verified byte-for-byte
// against a real `pyroaring.BitMap.serialize()` output; the two magic-cookie
// constants below were WRONG (0x3B4D/0x373B) before G2ICE-80 and could never
// have decoded a real externally-produced bitmap — see the ticket for the
// verification trail):
//   [u32 cookie]   either SERIAL_COOKIE_NO_RUNCONTAINER (12346 / 0x303A, no
//                  runs) or (SERIAL_COOKIE=12347/0x303B | ((n_containers-1)
//                  <<16)) when a run-flag bitset is present.
//   if cookie has run flag: [ceil(n/8) bytes] run-flag bitset, bit i set ⇒
//                  container i is a Run container.
//   [u32 size]     when cookie == NO_RUNCONTAINER (n_containers).
//   keyscards[n]:  n × { u16 key (high 16 bits), u16 cardinality-1 }.
//   offsets[n]:    u32 byte offset (from the start of THIS bitmap's bytes)
//                  of each container's payload. ALWAYS present for
//                  NO_RUNCONTAINER; present for the run-cookie form only
//                  when n >= NO_OFFSET_THRESHOLD(4) — getting this condition
//                  wrong (skipping it for NO_RUNCONTAINER when n<4, the
//                  overwhelmingly common shape for a single-file deletion
//                  vector) was the second pre-G2ICE-80 decode bug.
//   payloads in container order:
//       Array  : (card) × u16 sorted values.
//       Bitset : 1024 × u64 words (8192 bytes), raw little-endian words.
//       Run    : [u16 n_runs] then n_runs × { u16 start, u16 length-1 }.
//   A container is Array when card<=4096, else Bitset (the writer here never
//   emits Run containers — valid per spec, just not maximally compact).
//
// 64-bit portable format (CRoaring "frozen"/portable r64, i.e. RoaringTreemap
// in roaring-rs, which is what delta-kernel's DV reader deserialises):
//   [u64 n_buckets] then n_buckets × { u32 high32 key, <32-bit portable map> }.
//   `roaring_deserialize_r64` parses that; the 32-bit reader handles one map.
//   Only n_buckets<=1 is supported (read AND write) — a Delta/Iceberg
//   deletion vector addresses rows within one physical file, always < 2^32,
//   so every value naturally lands in bucket 0; a value needing a second
//   bucket cannot arise from a `uint32_t` input array by construction.
//
// Tiger Style: PODs, ≥2 asserts/fn, bounded loops, no exceptions, Arena allocs.

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "bolt/bolt_arena.h"

namespace bolt {
namespace ingest {

// Cookies + thresholds from the CRoaring portable spec (RoaringFormatSpec /
// CRoaring's roaring_array.h enum; ground-truthed against a real
// `pyroaring.BitMap.serialize()` byte dump — see the .h banner above).
static constexpr uint32_t kRoaringCookieNoRun  = 12346u;   // 0x303A
static constexpr uint32_t kRoaringCookieRun     = 12347u;   // 0x303B
static constexpr uint32_t kRoaringNoOffsetThr   = 4u;
// 32-bit portable cookie callers may probe (low 16 bits when run-flag set).
static constexpr uint32_t kRoaringSerialCookie  = 12347u;

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

// Write every set value, ascending, into `dst[0..dst_cap)`. Returns false
// (leaving *out_n untouched) if dst_cap < cardinality. Used by callers that
// need to UNION an existing bitmap with new values (e.g. merging a Delta
// deletion vector's prior contents with newly-deleted rows) — the decoder
// above has no other way to enumerate a bitmap's members.
bool roaring_to_sorted_array(const RoaringBitmap* bm, uint32_t* dst,
                             uint64_t dst_cap, uint64_t* out_n) noexcept;

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

// ---------------------------------------------------------------------------
// Serialisation (write path — G2ICE-80). `sorted_unique_values` MUST be
// strictly increasing (caller sorts+dedupes first; verified cheaply here and
// rejected otherwise, never silently reordered). Emits the NO_RUNCONTAINER
// form only (no run containers) — valid per spec, matches every decoder
// above, just not maximally compact for long consecutive runs.
// ---------------------------------------------------------------------------

// Upper bound on the serialised byte size for `n_values` sorted values.
// Always safe to over-allocate against; `roaring_serialize` reports the
// exact length actually used.
uint64_t roaring_serialize_bound(uint64_t n_values) noexcept;

// Serialise into caller-owned `dst[0..dst_cap)`. `scratch` holds the bounded
// per-container bookkeeping array (<=65536 entries) — never the arena `dst`
// came from, so the caller may reset scratch itself after this call.
// Returns false (and *out_len=0) on a too-small `dst_cap`, an unsorted /
// duplicate input, or a null argument.
bool roaring_serialize(const uint32_t* sorted_unique_values, uint64_t n_values,
                       Arena* scratch, uint8_t* dst, uint64_t dst_cap,
                       uint64_t* out_len) noexcept;

// 64-bit treemap wrapper (matches `roaring_deserialize_r64` / what
// delta-kernel's `RoaringTreemap::deserialize_from` expects): a single
// bucket (high32=0) wrapping one `roaring_serialize` payload — sufficient
// for any `uint32_t` value array by construction (see the r64 format note
// above). `n_values==0` emits an empty (0-bucket) treemap.
uint64_t roaring_serialize_r64_bound(uint64_t n_values) noexcept;
bool roaring_serialize_r64_single_bucket(const uint32_t* sorted_unique_values,
                                         uint64_t n_values, Arena* scratch,
                                         uint8_t* dst, uint64_t dst_cap,
                                         uint64_t* out_len) noexcept;

}  // namespace ingest
}  // namespace bolt
