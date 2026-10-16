// Fixed-extent, page-aware native event ring. Link bolt::io_file (BOLT_BUILD_IO).
// No database or xchannel runtime.
#pragma once

#include <cstdint>

namespace bolt { class Budget; }
namespace bolt::io {

enum class RingStatus : std::uint8_t {
    kOk, kEmpty, kFull, kBusy, kBadArg, kNoMemory, kIo, kCorrupt,
    kGap, kSequenceEnd, kUnsupported, kDecodeRefused,
};

struct MappedRing;
struct RingOptions {
    std::uint64_t capacity_bytes;
    std::uint64_t max_record_bytes;
    std::uint32_t readers;
    bool delete_on_close;
    bool prefault;
    // Opt-in flush retention: unflushed pages fence wrap until explicit flush.
    // False is an ephemeral ring; after unflushed physical reuse, a contiguous
    // durable prefix can no longer be certified and flush returns kUnsupported.
    bool retain_until_flush;
    // Borrowed byte/count hierarchies must outlive the ring. Reserve once at
    // open; no per-record Budget operations. Null disables that admission.
    bolt::Budget* disk_budget;
    bolt::Budget* memory_budget;
    bolt::Budget* file_budget;
};

struct RingReservation {
    std::uint8_t* data;
    std::uint64_t bytes;
    std::uint64_t position;
    std::uint64_t sequence;
    std::uint64_t token;
};

struct RingView {
    const std::uint8_t* data;
    std::uint64_t bytes;
    std::uint64_t position;
    std::uint64_t sequence;
    std::uint64_t lease;
    // kGap carries [lost_begin, lost_end), in whole-event sequence units.
    std::uint64_t lost_begin;
    std::uint64_t lost_end;
    std::uint32_t reader;
};

struct RingSnapshot {
    std::uint64_t capacity_bytes;
    std::uint64_t active_limit_bytes;
    std::uint64_t committed_position;
    std::uint64_t durable_position;
    std::uint64_t next_sequence;
    std::uint32_t page_bytes;
    std::uint32_t mapping_granularity;
};

// Cold open: NEW path only, physical preallocation, one fixed RW mapping.
// Capacity is a multiple of platform mapping/page granularity. Reader count
// is bounded by kMappedRingMaxReaders in bolt_io_limits.h. Every configured
// reader starts at event sequence 1, cursor 0; all participate in retention.
// Record spans are 64-byte aligned, packed across OS pages, and never split
// across the physical end (an explicit padding frame handles wrap). Geometry
// must allow two maximum spans plus one page, guaranteeing empty-ring progress.
// A record can span multiple pages and is published only as a complete event.
// This first owner API is in-process, ephemeral/replay storage: it provides no
// independent-process attachment or restart recovery/durable acknowledgement
// protocol. Explicit flush below reports an owner-lifetime durable byte prefix.
RingStatus mapped_ring_open(const char* path, const RingOptions& options,
                            MappedRing** out) noexcept;
// Caller quiesces all roles. Refuses kBusy for any live reservation/view and
// leaves the ring intact; successful close nulls *ring and releases budgets.
RingStatus mapped_ring_close(MappedRing** ring) noexcept;

// Single producer role: reserve/commit/cancel/set_limit/flush are serialized.
// Reserve exact event bytes, fill borrowed memory, then commit or cancel.
// No allocations, syscalls, file creation or remapping in these three calls.
// Lossless reserve returns kFull without discarding any unread event.
RingStatus mapped_ring_reserve(MappedRing*, std::uint64_t bytes,
                               RingReservation* out) noexcept;
// Explicit lossy admission: when full, may discard unread committed events
// for idle readers. Every affected reader receives kGap with its lost sequence
// interval before any new data. A live view always fences overwrite. Refusal
// does not evict; successful reserve may evict even if later cancelled.
RingStatus mapped_ring_reserve_overwrite(MappedRing*, std::uint64_t bytes,
                                         RingReservation* out) noexcept;
RingStatus mapped_ring_commit(MappedRing*, const RingReservation&) noexcept;
RingStatus mapped_ring_cancel(MappedRing*, const RingReservation&) noexcept;

// One consumer role per reader. kOk returns a pinned, CRC-checked borrowed
// view; release consumes exactly that event. kGap/kEmpty return no live view.
// Payload and pin remain valid until release. Tokens reject stale/double release.
RingStatus mapped_ring_peek(MappedRing*, std::uint32_t reader,
                            RingView* out) noexcept;
RingStatus mapped_ring_release(MappedRing*, const RingView&) noexcept;
// Release the pin WITHOUT consuming the event (e.g. retry descriptor admission).
// The next peek gets a new lease token; stale views never release that new pin.
RingStatus mapped_ring_unpin(MappedRing*, const RingView&) noexcept;

// Change logical retention cap within admitted physical capacity, page aligned.
// Refuse a shrink below current unread/pinned/reserved page commitment. This
// never grows or remaps physical storage or changes borrowed Budget charges.
RingStatus mapped_ring_set_limit(MappedRing*, std::uint64_t bytes) noexcept;
// Explicit blocking cold/IO operation, serialized with producer operations.
// Flush only the published dirty page ranges, then file-sync; update durable
// prefix only after all succeed. With retain_until_flush, unflushed committed
// data fences reuse and full returns kFull until explicit flush. In ephemeral
// mode flush refuses kUnsupported after unflushed physical reuse; it never
// certifies a prefix with lost unsynced bytes.
RingStatus mapped_ring_flush(MappedRing*) noexcept;
// Opt into ephemeral reuse: relinquish the unsynced prefix as a retention pin.
// Does not advance durable_position or claim crash persistence. Call between
// producer operations; subsequent dirty writes remain tracked for flush.
RingStatus mapped_ring_discard_unflushed(MappedRing*) noexcept;
// Producer/control role only; not an atomic cross-field snapshot for observers.
RingSnapshot mapped_ring_snapshot(const MappedRing*) noexcept;

} // namespace bolt::io
