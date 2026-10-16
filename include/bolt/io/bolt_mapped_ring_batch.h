// One native frame per whole ring event. No callback or terminal semantics.
#pragma once
#include "bolt/io/bolt_mapped_ring.h"
#include "bolt/wire/bolt_wire_frame.h"

namespace bolt::io {
namespace mapped_ring_batch_detail {
inline void publish_readonly_batch(const BoltBatch& source, std::uint64_t sequence, BoltBatch* out) noexcept {
    assert(out != nullptr);
    assert(source.dirty_mask == nullptr && source.dirty_words == 0);
    BoltBatch::init_empty(out);
    out->columns[0] = source.columns[0];
    out->columns[1] = source.columns[1];
    out->num_rows = source.num_rows;
    out->num_cols = source.num_cols;
    out->schema = source.schema;
    out->arena = source.arena;
    // Reused output shells must invalidate downstream pointer/generation caches.
    out->generation[0] = sequence;
    out->generation[1] = sequence;
    out->read_epoch.store(source.read_epoch.load(std::memory_order_relaxed), std::memory_order_relaxed);
    out->write_epoch.store(source.write_epoch.load(std::memory_order_relaxed), std::memory_order_relaxed);
}
} // namespace mapped_ring_batch_detail
// Source batch and all transitive buffers remain immutable/disjoint from the
// writable reservation until return. kFull leaves the caller batch pending.
// Direct encoding uses no temporary frame buffer. Caller chooses frame flags,
// kind and CRC seed; this helper invents no EOF/control convention.
inline RingStatus mapped_ring_write_batch(MappedRing* ring, const BoltBatch* batch,
        const wire::FrameMeta& meta, bool overwrite=false) noexcept {
    if (!ring || !batch || batch->num_cols > wire::kWireMaxCols ||
        batch->num_rows < 0 || batch->num_rows > UINT32_MAX) return RingStatus::kBadArg;
    assert(batch != nullptr);
    const auto bytes = wire::frame_size(batch, false);
    if (!bytes) return RingStatus::kUnsupported;
    RingReservation reservation{};
    auto status = overwrite ? mapped_ring_reserve_overwrite(ring, bytes, &reservation) :
                              mapped_ring_reserve(ring, bytes, &reservation);
    if (status != RingStatus::kOk) return status;
    wire::FrameStatus encoded{};
    const auto written = wire::frame_write(batch, meta, wire::FrameZones::kNone,
        nullptr, nullptr, reservation.data, static_cast<std::size_t>(reservation.bytes), &encoded);
    if (encoded != wire::FrameStatus::kOk || written != bytes) {
        const auto cancelled = mapped_ring_cancel(ring, reservation);
        return cancelled == RingStatus::kOk ? RingStatus::kUnsupported : cancelled;
    }
    assert(written == reservation.bytes);
    return mapped_ring_commit(ring, reservation);
}

// kOk returns a borrowed batch and a live explicit lease: release consumes the
// event; unpin permits retry. Both payload and descriptor arena must outlive
// every batch borrower. Both batch generations equal the ring event sequence,
// including repeat peeks after unpin. Do not reset/reuse that arena while any view uses it.
// Arena descriptors may allocate unless the caller prewarms sufficient storage
// and disables growth. Decode failure may advance arena scratch, but preserves
// outputs and unpins WITHOUT consuming; the same event remains pending.
// kDecodeRefused means native codec or descriptor admission refused; retry may
// succeed after scratch reconfiguration. The bool codec API cannot identify
// allocation failure separately from unsupported/malformed native layouts.
// Output objects, descriptor arena and its backing must be disjoint from each
// other and mapped payload storage. Caller exclusively owns the arena.
// kGap publishes only the gap token to lease (no live pin); other failures leave
// outputs unchanged. crc_seed must match the producer's FrameMeta::crc_seed.
inline RingStatus mapped_ring_peek_batch(MappedRing* ring, std::uint32_t reader,
        Arena* arena, BoltBatch* out, RingView* lease, wire::FrameView* frame=nullptr,
        std::uint32_t crc_seed=0) noexcept {
    if (!ring || !arena || !out || !lease) return RingStatus::kBadArg;
    assert(arena != nullptr);
    RingView candidate{};
    auto status = mapped_ring_peek(ring, reader, &candidate);
    if (status == RingStatus::kGap) { *lease = candidate; return status; }
    if (status != RingStatus::kOk) return status;
    wire::FrameView parsed{};
    BoltBatch batch;
    BoltBatch::init_empty(&batch);
    RingStatus decoded = RingStatus::kOk;
    if (candidate.bytes > SIZE_MAX ||
        wire::frame_parse(candidate.data, static_cast<std::size_t>(candidate.bytes), true,
                          &parsed, crc_seed) != wire::FrameStatus::kOk ||
        parsed.frame_len != candidate.bytes) decoded = RingStatus::kCorrupt;
    else if (!wire::bolt_wire_view(parsed.payload, parsed.payload_len, &batch, arena))
        decoded = RingStatus::kDecodeRefused;
    else if (batch.num_rows != parsed.header.rows) decoded = RingStatus::kCorrupt;
    if (decoded != RingStatus::kOk) {
        const auto unpinned = mapped_ring_unpin(ring, candidate);
        return unpinned == RingStatus::kOk ? decoded : unpinned;
    }
    assert(parsed.frame_len == candidate.bytes);
    mapped_ring_batch_detail::publish_readonly_batch(batch, candidate.sequence, out);
    *lease = candidate;
    if (frame) *frame = parsed;
    return RingStatus::kOk;
}
} // namespace bolt::io
