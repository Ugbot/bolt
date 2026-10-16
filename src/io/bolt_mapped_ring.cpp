#include "bolt/io/bolt_mapped_ring.h"
#include "bolt/io/bolt_file_io.h"
#include "bolt/io/bolt_crc32c.h"
#include "bolt/bolt_budget.h"

#include <atomic>
#include <cassert>
#include <cstring>
#include <new>

namespace bolt::io {
namespace {
constexpr std::uint64_t kRecordAlign = 64;
enum : std::uint32_t { kIdle, kReading, kEvicting };
enum : std::uint32_t { kData = 1, kPadding = 2 };
struct Record {
    std::uint64_t position, sequence, span, bytes;
    std::uint32_t kind, crc;
    std::uint64_t reserved[3];
};
static_assert(sizeof(Record) == kRecordAlign);
struct alignas(64) Reader {
    std::atomic<std::uint64_t> cursor{0};
    std::atomic<std::uint32_t> state{kIdle};
    std::uint64_t next_sequence = 1, lease = 0, lease_end = 0, lease_bytes = 0;
    std::uint64_t lost_begin = 0, lost_end = 0;
};
std::uint64_t span_for(std::uint64_t bytes) noexcept {
    assert(bytes <= kFileReadMaxBytes);
    const auto span = (bytes + sizeof(Record) + kRecordAlign - 1) & ~(kRecordAlign - 1);
    assert(span >= sizeof(Record));
    return span;
}
RingStatus io_status(FileIoStatus s) noexcept {
    if (s == FileIoStatus::kOk) return RingStatus::kOk;
    if (s == FileIoStatus::kNoMemory) return RingStatus::kNoMemory;
    if (s == FileIoStatus::kUnsupported) return RingStatus::kUnsupported;
    if (s == FileIoStatus::kBadArg) return RingStatus::kBadArg;
    return RingStatus::kIo;
}
} // namespace

struct alignas(64) MappedRing {
    WritableMappedFile map{};
    RingOptions options{};
    alignas(64) std::atomic<std::uint64_t> visible{0};
    std::uint64_t position = 0, next_sequence = 1, next_token = 1;
    std::uint64_t durable = 0, reclaim_floor = 0, limit = 0;
    std::uint64_t reservation_end = 0, reservation_span = 0;
    RingReservation reservation{};
    bool reserved = false, durable_gap = false;
    Reader readers[kMappedRingMaxReaders];
};

namespace {
std::uint64_t page_floor(const MappedRing* r, std::uint64_t p) noexcept {
    assert(r != nullptr && r->map.page != 0);
    assert((r->map.page & (r->map.page - 1)) == 0);
    return p & ~(std::uint64_t{r->map.page} - 1);
}
std::uint64_t refresh_floor(MappedRing* r) noexcept {
    assert(r != nullptr && r->options.readers <= kMappedRingMaxReaders);
    std::uint64_t minimum = r->position;
    for (std::uint32_t i = 0; i < r->options.readers; ++i) {
        const auto p = r->readers[i].cursor.load(std::memory_order_acquire);
        if (p < minimum) minimum = p;
    }
    if (r->options.retain_until_flush && r->durable < minimum) minimum = r->durable;
    const auto floor = page_floor(r, minimum);
    assert(floor >= r->reclaim_floor);
    r->reclaim_floor = floor;
    return floor;
}
void uncharge(const RingOptions& o, std::uint32_t admitted) noexcept {
    assert(admitted <= 3);
    const auto memory = o.capacity_bytes + sizeof(MappedRing);
    if (admitted >= 3 && o.memory_budget) o.memory_budget->release(memory);
    if (admitted >= 2 && o.disk_budget) o.disk_budget->release(o.capacity_bytes);
    if (admitted >= 1 && o.file_budget) o.file_budget->release(1);
    assert(memory >= o.capacity_bytes);
}
bool charge(const RingOptions& o) noexcept {
    assert(o.capacity_bytes <= kFileReadMaxBytes);
    const auto memory = o.capacity_bytes + sizeof(MappedRing);
    assert(memory > o.capacity_bytes);
    if (o.file_budget && !o.file_budget->try_reserve(1)) return false;
    if (o.disk_budget && !o.disk_budget->try_reserve(o.capacity_bytes)) {
        uncharge(o, 1); return false;
    }
    if (o.memory_budget && !o.memory_budget->try_reserve(memory)) {
        uncharge(o, 2); return false;
    }
    return true;
}
bool geometry(const RingOptions& o) noexcept {
    const std::uint64_t page = io_page_size(), granularity = io_map_granularity();
    if (!page || !granularity || (page & (page - 1)) || (granularity & (granularity - 1))) return false;
    assert(page % kRecordAlign == 0);
    const auto unit = page > granularity ? page : granularity;
    if (o.capacity_bytes > kFileReadMaxBytes || o.capacity_bytes > SIZE_MAX ||
        !o.capacity_bytes || o.capacity_bytes % unit || o.max_record_bytes > kFileReadMaxBytes ||
        !o.readers || o.readers > kMappedRingMaxReaders) return false;
    assert(o.capacity_bytes >= page);
    return 2 * span_for(o.max_record_bytes) + page <= o.capacity_bytes;
}
std::uint32_t record_crc(Record header, const std::uint8_t* data) noexcept {
    assert(header.bytes <= kFileReadMaxBytes);
    assert(header.bytes == 0 || data != nullptr);
    header.crc = 0;
    const auto seed = crc32c(reinterpret_cast<const std::uint8_t*>(&header), sizeof(header));
    return header.bytes ? crc32c(data, static_cast<std::size_t>(header.bytes), seed) : seed;
}
bool same_reservation(const MappedRing* r, const RingReservation& x) noexcept {
    assert(r != nullptr);
    assert(!r->reserved || r->reservation.data != nullptr);
    const auto& current = r->reservation;
    return r->reserved && x.token == current.token && x.position == current.position &&
        x.sequence == current.sequence && x.bytes == current.bytes && x.data == current.data;
}
void unlock_readers(MappedRing* r, std::uint32_t count) noexcept {
    assert(r != nullptr && count <= r->options.readers);
    for (std::uint32_t i = 0; i < count; ++i) {
        assert(r->readers[i].state.load(std::memory_order_relaxed) == kEvicting);
        r->readers[i].state.store(kIdle, std::memory_order_release);
    }
}
bool evict_idle(MappedRing* r, std::uint64_t end) noexcept {
    assert(r != nullptr && !r->reserved);
    auto floor = page_floor(r, r->position);
    if (r->options.retain_until_flush && r->durable < floor) floor = page_floor(r, r->durable);
    if (end - floor > r->limit) return false;
    for (std::uint32_t i = 0; i < r->options.readers; ++i) {
        auto expected = std::uint32_t{kIdle};
        if (!r->readers[i].state.compare_exchange_strong(expected, kEvicting, std::memory_order_acquire)) {
            unlock_readers(r, i); return false;
        }
    }
    for (std::uint32_t i = 0; i < r->options.readers; ++i) {
        auto& reader = r->readers[i];
        if (reader.next_sequence < r->next_sequence) {
            if (reader.lost_begin == 0) reader.lost_begin = reader.next_sequence;
            reader.lost_end = r->next_sequence;
            reader.next_sequence = r->next_sequence;
        }
        reader.cursor.store(r->position, std::memory_order_release);
    }
    r->reclaim_floor = floor;
    unlock_readers(r, r->options.readers);
    return true;
}
template<bool Overwrite>
RingStatus reserve(MappedRing* r, std::uint64_t bytes, RingReservation* out) noexcept {
    if (out == nullptr) return RingStatus::kBadArg;
    *out = {};
    if (r == nullptr || bytes > r->options.max_record_bytes) return RingStatus::kBadArg;
    assert(r->map.base != nullptr && r->position % kRecordAlign == 0);
    if (r->reserved) return RingStatus::kBusy;
    if (r->next_sequence == UINT64_MAX || r->next_token == UINT64_MAX) return RingStatus::kSequenceEnd;
    const auto span = span_for(bytes), offset = r->position % r->map.size;
    const auto padding = span > r->map.size - offset ? r->map.size - offset : 0;
    if (r->position > UINT64_MAX - padding - span) return RingStatus::kSequenceEnd;
    const auto end = r->position + padding + span;
    if (end - r->reclaim_floor > r->limit && end - refresh_floor(r) > r->limit) {
        if constexpr (!Overwrite) return RingStatus::kFull;
        else if (!evict_idle(r, end)) return RingStatus::kFull;
    }
    assert(end - r->reclaim_floor <= r->limit);
    if (end - r->durable > r->map.size) r->durable_gap = true;
    r->reservation = {r->map.base + ((r->position + padding) % r->map.size) + sizeof(Record),
                      bytes, r->position + padding, r->next_sequence, r->next_token++};
    r->reservation_end = end; r->reservation_span = span; r->reserved = true;
    *out = r->reservation;
    return RingStatus::kOk;
}
} // namespace

RingStatus mapped_ring_open(const char* path, const RingOptions& o, MappedRing** out) noexcept {
    if (!out) return RingStatus::kBadArg;
    *out = nullptr;
    if (!path || !path[0] || !geometry(o)) return RingStatus::kBadArg;
    assert(o.capacity_bytes <= kFileReadMaxBytes);
    if (!charge(o)) return RingStatus::kFull;
    auto* storage = io_alloc_aligned(sizeof(MappedRing), alignof(MappedRing));
    if (!storage) { uncharge(o, 3); return RingStatus::kNoMemory; }
    auto* r = new (storage) MappedRing{};
    const auto status = mapped_write_create(path, o.capacity_bytes, o.delete_on_close, o.prefault, &r->map);
    if (status != FileIoStatus::kOk) {
        r->~MappedRing(); io_free_aligned(storage); uncharge(o, 3); return io_status(status);
    }
    r->options = o; r->limit = o.capacity_bytes;
    assert(r->map.size == o.capacity_bytes);
    *out = r;
    return RingStatus::kOk;
}
RingStatus mapped_ring_close(MappedRing** ptr) noexcept {
    if (!ptr) return RingStatus::kBadArg;
    auto* r = *ptr;
    if (!r) return RingStatus::kOk;
    assert(r->map.base != nullptr);
    if (r->reserved) return RingStatus::kBusy;
    for (std::uint32_t i = 0; i < r->options.readers; ++i)
        if (r->readers[i].state.load(std::memory_order_acquire) != kIdle) return RingStatus::kBusy;
    const auto o = r->options;
    mapped_write_close(&r->map); r->~MappedRing();
    io_free_aligned(reinterpret_cast<std::uint8_t*>(r)); uncharge(o, 3); *ptr = nullptr;
    assert(*ptr == nullptr);
    return RingStatus::kOk;
}
RingStatus mapped_ring_reserve(MappedRing* r, std::uint64_t n, RingReservation* out) noexcept {
    return reserve<false>(r, n, out);
}
RingStatus mapped_ring_reserve_overwrite(MappedRing* r, std::uint64_t n, RingReservation* out) noexcept {
    return reserve<true>(r, n, out);
}
RingStatus mapped_ring_commit(MappedRing* r, const RingReservation& x) noexcept {
    if (!r || !same_reservation(r, x)) return RingStatus::kBadArg;
    assert(x.position >= r->position && r->reservation_end > x.position);
    if (x.position != r->position) {
        Record padding{r->position, 0, x.position - r->position, 0, kPadding, 0, {}};
        padding.crc = record_crc(padding, nullptr);
        std::memcpy(r->map.base + (r->position % r->map.size), &padding, sizeof(padding));
    }
    Record header{x.position, x.sequence, r->reservation_span, x.bytes, kData, 0, {}};
    header.crc = record_crc(header, x.data);
    std::memcpy(x.data - sizeof(Record), &header, sizeof(header));
    r->position = r->reservation_end; ++r->next_sequence; r->reserved = false;
    r->visible.store(r->position, std::memory_order_release);
    assert(r->position == r->reservation_end);
    return RingStatus::kOk;
}
RingStatus mapped_ring_cancel(MappedRing* r, const RingReservation& x) noexcept {
    if (!r || !same_reservation(r, x)) return RingStatus::kBadArg;
    assert(r->visible.load(std::memory_order_relaxed) == r->position);
    r->reserved = false;
    assert(!r->reserved);
    return RingStatus::kOk;
}

namespace {
bool same_view(const MappedRing* r, const RingView& view) noexcept {
    assert(r != nullptr);
    assert(r->options.readers <= kMappedRingMaxReaders);
    if (view.reader >= r->options.readers) return false;
    const auto& reader = r->readers[view.reader];
    return reader.state.load(std::memory_order_relaxed) == kReading && reader.lease == view.lease &&
        reader.cursor.load(std::memory_order_relaxed) == view.position &&
        reader.next_sequence == view.sequence && reader.lease_bytes == view.bytes &&
        view.data == r->map.base + (view.position % r->map.size) + sizeof(Record);
}
RingStatus read_record(MappedRing* r, Reader& reader, std::uint32_t id, RingView* out) noexcept {
    assert(r != nullptr && out != nullptr);
    assert(reader.state.load(std::memory_order_relaxed) == kReading);
    if (reader.lost_begin) {
        out->reader = id; out->lost_begin = reader.lost_begin; out->lost_end = reader.lost_end;
        reader.lost_begin = reader.lost_end = 0; return RingStatus::kGap;
    }
    const auto visible = r->visible.load(std::memory_order_acquire);
    auto position = reader.cursor.load(std::memory_order_relaxed);
    for (std::uint32_t attempt = 0; attempt < 2; ++attempt) {
        if (position == visible) return RingStatus::kEmpty;
        if (position > visible || visible - position > r->map.size) return RingStatus::kCorrupt;
        const auto offset = position % r->map.size;
        Record h{}; std::memcpy(&h, r->map.base + offset, sizeof(h));
        if (h.position != position || h.span < sizeof(h) || h.span % kRecordAlign ||
            h.span > r->map.size - offset || h.span > visible - position) return RingStatus::kCorrupt;
        if (h.kind == kPadding) {
            if (h.bytes || h.sequence || h.span != r->map.size - offset ||
                h.crc != record_crc(h, nullptr)) return RingStatus::kCorrupt;
            position += h.span; reader.cursor.store(position, std::memory_order_release); continue;
        }
        if (h.kind != kData || h.sequence != reader.next_sequence || h.bytes > r->options.max_record_bytes ||
            span_for(h.bytes) != h.span || h.crc != record_crc(h, r->map.base + offset + sizeof(h)))
            return RingStatus::kCorrupt;
        if (reader.lease == UINT64_MAX) return RingStatus::kSequenceEnd;
        ++reader.lease; reader.lease_end = position + h.span; reader.lease_bytes = h.bytes;
        *out = {r->map.base + offset + sizeof(h), h.bytes, position, h.sequence, reader.lease, 0, 0, id};
        return RingStatus::kOk;
    }
    return RingStatus::kCorrupt;
}
} // namespace

RingStatus mapped_ring_peek(MappedRing* r, std::uint32_t id, RingView* out) noexcept {
    if (!out) return RingStatus::kBadArg;
    *out = {};
    if (!r || id >= r->options.readers) return RingStatus::kBadArg;
    auto& reader = r->readers[id]; auto expected = std::uint32_t{kIdle};
    if (!reader.state.compare_exchange_strong(expected, kReading, std::memory_order_acquire)) return RingStatus::kBusy;
    assert(reader.cursor.load(std::memory_order_relaxed) % kRecordAlign == 0);
    const auto status = read_record(r, reader, id, out);
    if (status != RingStatus::kOk) reader.state.store(kIdle, std::memory_order_release);
    assert(status != RingStatus::kOk || out->data != nullptr);
    return status;
}
RingStatus mapped_ring_release(MappedRing* r, const RingView& view) noexcept {
    if (!r || !same_view(r, view)) return RingStatus::kBadArg;
    auto& reader = r->readers[view.reader];
    assert(reader.lease_end > view.position);
    assert(reader.lease_end <= r->visible.load(std::memory_order_acquire));
    ++reader.next_sequence;
    reader.cursor.store(reader.lease_end, std::memory_order_release);
    reader.state.store(kIdle, std::memory_order_release);
    return RingStatus::kOk;
}
RingStatus mapped_ring_unpin(MappedRing* r, const RingView& view) noexcept {
    if (!r || !same_view(r, view)) return RingStatus::kBadArg;
    auto& reader = r->readers[view.reader];
    assert(reader.lease_end > view.position);
    assert(reader.cursor.load(std::memory_order_relaxed) == view.position);
    reader.state.store(kIdle, std::memory_order_release);
    return RingStatus::kOk;
}
RingStatus mapped_ring_set_limit(MappedRing* r, std::uint64_t bytes) noexcept {
    if (!r || bytes > r->map.size || !bytes || bytes % r->map.page ||
        2 * span_for(r->options.max_record_bytes) + r->map.page > bytes) return RingStatus::kBadArg;
    assert(r->limit <= r->map.size);
    const auto end = r->reserved ? r->reservation_end : r->position;
    if (end - refresh_floor(r) > bytes) return RingStatus::kFull;
    r->limit = bytes;
    assert(r->limit == bytes);
    return RingStatus::kOk;
}
RingStatus mapped_ring_flush(MappedRing* r) noexcept {
    if (!r) return RingStatus::kBadArg;
    assert(r->durable <= r->position);
    if (r->reserved) return RingStatus::kBusy;
    if (r->durable_gap) return RingStatus::kUnsupported;
    const auto bytes = r->position - r->durable, off = r->durable % r->map.size;
    const auto first = bytes < r->map.size - off ? bytes : r->map.size - off;
    auto status = mapped_write_flush(&r->map, off, first);
    if (status == FileIoStatus::kOk && first < bytes) status = mapped_write_flush(&r->map, 0, bytes - first);
    if (status != FileIoStatus::kOk) return io_status(status);
    r->durable = r->position;
    assert(r->durable == r->position);
    return RingStatus::kOk;
}
RingStatus mapped_ring_discard_unflushed(MappedRing* r) noexcept {
    if (!r) return RingStatus::kBadArg;
    assert(r->durable <= r->position);
    if (r->reserved) return RingStatus::kBusy;
    r->options.retain_until_flush = false;
    assert(!r->options.retain_until_flush);
    return RingStatus::kOk;
}
RingSnapshot mapped_ring_snapshot(const MappedRing* r) noexcept {
    if (!r) return {};
    assert(r->map.size != 0);
    assert(r->durable <= r->position);
    return {r->map.size, r->limit, r->position, r->durable, r->next_sequence,
            r->map.page, r->map.granularity};
}
} // namespace bolt::io
